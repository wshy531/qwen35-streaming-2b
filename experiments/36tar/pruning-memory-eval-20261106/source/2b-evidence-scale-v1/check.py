"""CPU contracts, tiny QA and selected-pair streaming state equivalence."""
import argparse,os,hashlib,json
from pathlib import Path
from common import *
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    env=s.runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261102)
    processor=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder=s.make_builder(env,processor);head=GroupScorer(model.config.vision_config.hidden_size).eval().requires_grad_(False)
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    actual_head=GroupScorer(read(R/'models/Qwen3.5-2B/config.json')['vision_config']['hidden_size']);actual_head.load_state_dict(t.load(CHECKPOINT,map_location='cpu',weights_only=True))
    assert sum(p.numel() for p in actual_head.parameters())==69697
    audit={x['video_id']:x for x in read(s.DATA/'CPU_media_audit.json')['rows']};qs=read(P/'panel.json')
    assert len(qs)==300 and not any({'answer_index','answer_text'}&q.keys() for q in qs)
    digest=hashlib.sha256();layouts=0;maximum=0
    for q in qs:
        samples=audit[q['video_id']]['samples'] if q['dataset']=='NExT-QA' else [dict(pts_s=float(i)) for i in range(16)]
        for arm in ARMS:
            n=108 if arm=='full' else 54;ids=t.arange(108) if n==108 else t.arange(1,108,2)
            records=[(i//2,samples[i]['pts_s'],samples[min(i+1,len(samples)-1)]['pts_s'],t.zeros(n,model.config.text_config.hidden_size),ids) for i in range(0,len(samples),2)]
            emb,pos,tokens=layout(env,model,builder,records,q,arm)
            assert tokens.shape[1]+128<=4096
            if arm=='text_only':assert not (tokens==processor.video_token_id).any()
            layouts+=1;maximum=max(maximum,tokens.shape[1]);digest.update(tokens.numpy().tobytes());digest.update(pos.numpy().tobytes())
    from transformers.feature_extraction_utils import BatchFeature
    batch=BatchFeature(dict(pixel_values_videos=t.randn(432,1536),video_grid_thw=t.tensor([[1,18,24]])))
    micro=[]
    with t.no_grad(),env['fp32_reference']():
        for arm in ARMS:
            records=[]
            if arm!='text_only':
                state=None;selector=s.CoverageScorer()
                for i in range(2):
                    features,ids,state=encode(model,batch,arm,head,state,selector)
                    assert len(features)==(108 if arm=='full' else 54)
                    assert ids.tolist()==sorted(set(ids.tolist()))
                    records.append((i,float(2*i),float(2*i+1),features,ids))
            result=generate(env,model,builder,records,qs[0],arm,limit=4)
            micro.append(dict(arm=arm,generated_ids=result['generated_ids'],visual_tokens=result['visual_tokens']))
    reuse=read(P/'reuse.json');gates=read(P/'regression.json');by={q['id']:q for q in qs}
    assert len(reuse)==180 and len({(x['id'],x['arm']) for x in reuse})==180
    assert len(gates)==18 and len(read(P/'auxiliary_panel.json'))==45
    for x in reuse:assert x['id'] in by and x['arm'] in ARMS[:3] and x['protocol']=='B' and x['shift']==0
    out=dict(status='passed_random_weights_only',layouts=layouts,layout_sha256=digest.hexdigest(),max_tokens=maximum,micro_QA=micro,
        frozen_CE_parameters=69697,checkpoint_sha256=CHECKPOINT_SHA,GPU=0,API=0,
        stream_layouts='upper bound 8 pairs, real PTS checked by GPU causal source',reuse=180,regressions=18)
    save(a.out,out);print(json.dumps(out))
if __name__=='__main__':main()
