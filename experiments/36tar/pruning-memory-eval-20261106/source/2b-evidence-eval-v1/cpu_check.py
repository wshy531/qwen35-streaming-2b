import argparse,os
from gpu_profile import *
from response import question
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261102)
    processor=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder=make_builder(env,processor);audit={x['video_id']:x for x in read(DATA/'CPU_media_audit.json')['rows']}
    layouts=0;maximum=0;digest=hashlib.sha256();vision=0
    for q in read(P/'profile.json'):
        samples=audit[q['video_id']]['samples'];vision+=2*((len(samples)+1)//2)
        for n in (108,54):
            ids=t.arange(108) if n==108 else t.arange(1,108,2);records=[]
            for i in range(0,len(samples),2):records.append((i//2,samples[i]['pts_s'],samples[min(i+1,len(samples)-1)]['pts_s'],t.zeros(n,model.config.text_config.hidden_size),ids))
            for protocol in ('A','B'):
                _,pos,tokens=context(model,builder,records,question(q,0,protocol))
                assert tokens.shape[1]+(8 if protocol=='A' else 128)<=4096
                maximum=max(maximum,tokens.shape[1]);layouts+=1;digest.update(tokens.numpy().tobytes());digest.update(pos.numpy().tobytes())
    from transformers.feature_extraction_utils import BatchFeature
    batch=BatchFeature(dict(pixel_values_videos=t.randn(432,1536),video_grid_thw=t.tensor([[1,18,24]])))
    qa=[];q=read(P/'profile.json')[0]
    with t.no_grad(),env['fp32_reference']():
        for arm in ('full','fixed12'):
            records=records_for(model,[(0.,.5,batch)],arm)
            for protocol in ('A','B'):
                # Keep full production limits to check generation exits by all EOS or cap.
                result=generate(env,model,builder,records,q,protocol);qa.append(dict(arm=arm,protocol=protocol,generated=len(result['generated_ids'])))
    out=dict(status='passed_random_weights',layouts=layouts,layout_sha256=digest.hexdigest(),max_input_tokens=maximum,vision_calls=vision,
        micro_QA=qa,GPU=0,API=0,model_loads=0)
    save(a.out,out);print(json.dumps(out))
if __name__=='__main__':main()
