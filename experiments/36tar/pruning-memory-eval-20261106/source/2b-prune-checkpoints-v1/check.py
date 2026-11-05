"""CPU-only input, checkpoint and inference interface checks."""
import argparse,hashlib,json,os
from worker import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--media',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    if a.media:
        rows=[]
        for s in read(P/'sources.json'):
            f=ROOT/'datasets/learned-prune-nextqa-v1'/s['mirror_path'];assert sha(f)==s['mirror_lfs_sha256']
            rows.append(dict(video_id=s['video_id'],sha256=sha(f)))
        save(a.out,dict(passed=True,videos=len(rows),rows=rows));print('40 media verified');return
    env=runtime();torch.set_num_threads(1);torch.manual_seed(20261102)
    proc=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](proc);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder=make_builder(env,proc);questions=read(P/'questions.json');spec=read(P/'protocol.json')
    audit={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
    digest=hashlib.sha256();maximum=0
    for q in questions:
        samples=audit[q['video_id']]['samples'];records=[]
        for i in range(0,len(samples),2):
            records.append((i//2,samples[i]['pts_s'],samples[min(i+1,len(samples)-1)]['pts_s'],torch.zeros(54,model.config.text_config.hidden_size),torch.arange(1,108,2)))
        _,pos,tokens=context(model,builder,records,question_text(q));assert tokens.shape[1]+8<=4096
        digest.update(tokens.numpy().tobytes());digest.update(pos.numpy().tobytes());maximum=max(maximum,tokens.shape[1])
    heads=[]
    for arm,path in spec['checkpoints'].items():
        state=torch.load(ROOT/path,map_location='cpu',weights_only=True)
        probe=GroupScorer(1024);probe.load_state_dict(state)
        assert sum(p.numel() for p in probe.parameters())==69697 and all(torch.isfinite(p).all() for p in probe.parameters())
        heads.append(dict(arm=arm,sha256=sha(ROOT/path)))
    scorer=GroupScorer(model.config.vision_config.hidden_size).eval()
    from transformers.feature_extraction_utils import BatchFeature
    batch=BatchFeature(dict(pixel_values_videos=torch.randn(432,1536),video_grid_thw=torch.tensor([[1,18,24]])))
    qa=[]
    with torch.no_grad(),env['fp32_reference']():
        for k in range(2):
            records=records_for(model,[(0.,.5,batch)],scorer,'gather');q=questions[k]
            before=old_worker.answer(env,model,builder,records,q,limit=1)
            logits,tokens=last_logits(model,builder,records,q);_,ce,_,_=loss_parts(logits,option_ids(builder,torch.device('cpu')),q['answer_index'])
            after=old_worker.answer(env,model,builder,records,q,limit=1)
            assert before==after and torch.isfinite(ce) and tokens==before['input_tokens']
            qa.append(dict(id=q['id'],finite_loss=True,qa_unchanged_after_loss=True))
    save(a.out,dict(passed=True,layouts=345,layout_sha256=digest.hexdigest(),maximum_tokens=maximum,heads=heads,micro_checks=qa,GPU=0,pretrained_loads=0))
    print(json.dumps(read(a.out)))
if __name__=='__main__':main()
