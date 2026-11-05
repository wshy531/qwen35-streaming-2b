import os,sys,json,argparse
from pathlib import Path
from types import SimpleNamespace
from development_worker import R,P,old,read
from aligned_encoder import encode
def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=old.runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
    processor=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    builder=old.make_builder(env,processor);selectors=[old.CoverageScorer() for _ in range(3)]
    import numpy as np
    rng=np.random.default_rng(20261103)
    with t.inference_mode(),env['fp32_reference']():
        for i in range(2):
            frames=[env['FramePacket'](frame_id=2*i+k,timestamp=float(2*i+k),rgb=rng.integers(0,256,(288,384,3),np.uint8),decoded_count=2*i+k+1) for k in (0,1)]
            batch,_=builder.pair(*frames,0,0)
            a,ai,_=old.encode_pair(model,batch,'heuristic11',None,None,selectors[0])
            b,bi,_=encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,selectors[1],merge=False)
            c,ci,meta=encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,selectors[2])
            assert t.equal(a,b) and t.equal(ai,bi) and t.equal(ai,ci)
            assert t.equal(selectors[0].previous,selectors[2].previous) and int(meta['mass'].sum())==108 and t.isfinite(c).all()
        audits={x['video_id']:x for x in read(P/'samples.json')};layouts=[]
        for q in read(P/'questions.json'):
            samples=audits[q['video_id']]['samples']
            for arm in read(P/'protocol.json')['arms']:
                ids=t.arange(108) if arm=='full' else t.arange(1,108,2)
                records=[(i//2,samples[i]['pts_s'],samples[min(i+1,len(samples)-1)]['pts_s'],t.zeros((len(ids),model.get_input_embeddings().embedding_dim)),ids) for i in range(0,len(samples),2)]
                for shift in (0,2):
                    emb,pos,token_ids=old.context(model,builder,records,old.question_text(old.rotate(q,shift)))
                    assert token_ids.shape[1]+8<=4096
                    layouts.append(dict(id=q['id'],arm=arm,shift=shift,tokens=token_ids.shape[1]))
        # Exercise the actual development answer/score interface for the new features.
        from development_worker import worker
        q=read(P/'questions.json')[0]
        answer=worker.answer(env,model,builder,[(0,0.,1.,c,ci)],q,limit=2)
        assert len(answer['generated_ids'])<=2
    deps=[]
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p:
            p=Path(p).resolve()
            if p.suffix=='.py' and p.is_relative_to(R.resolve()):
                name=p.relative_to(R.resolve()).as_posix()
                if name.startswith(('experiments/','scripts/','repo/','publish/')):deps.append(name)
    assert len(layouts)==2760
    with out.open('x',encoding='utf-8') as f:json.dump(dict(status='passed',layouts=layouts,old_heuristic11_exact=True,same_ids=True,tiny_vision_calls=6,tiny_QA=1,loaded_files=sorted(set(deps)),GPU=0,pretrained_loads=0),f,indent=2)
    print('2760 layouts, original layer11 equality, six tiny ViT forwards and QA passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
