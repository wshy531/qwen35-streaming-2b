"""Stateful RGB-pair -> five banks -> isolated QA, with random tiny weights."""
import os,json,argparse,sys
from pathlib import Path
from types import SimpleNamespace
from session import ROOT,Session,MODES
from sember_common import runtime
from resolution_runtime import make_builder
def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261101)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    builder=make_builder(env,processor,(288,384));session=Session(env,model,builder)
    import numpy as np
    rng=np.random.default_rng(20261101)
    with t.inference_mode():
        for i in range(9):
            a=SimpleNamespace(frame_id=2*i,timestamp=float(2*i),rgb=rng.integers(0,256,(288,384,3),np.uint8))
            b=SimpleNamespace(frame_id=2*i+1,timestamp=float(2*i+1),rgb=rng.integers(0,256,(288,384,3),np.uint8))
            session.append_pair(a,b)
            if i==0:
                from content_prune import encode
                from midprune import encode as old_encode
                inputs,_=builder.pair(a,b,0,0)
                for new,old in [('mid_fixed','mid_odd'),('post_fixed','post_odd')]:
                    x,ids,_=encode(model.model.visual,inputs.pixel_values_videos,inputs.video_grid_thw,mode=new)
                    y,other,_=old_encode(model.model.visual,inputs.pixel_values_videos,inputs.video_grid_thw,mode=old)
                    assert t.equal(x,y) and t.equal(ids,other)
        before=session.status();fingerprint=session.bank.selector_digest();rows=[]
        for m in MODES:
            assert len(session.bank.banks[m].nodes())==8
            r=session.ask(m,'What is visible? A. A cup. B. A book.',lambda item:None,max_new_tokens=2)
            assert r['visual_tokens']==(864 if m=='full' else 432) and r['branch_released'] and r['read_state_unchanged']
            rows.append(dict(mode=m,visual_tokens=r['visual_tokens'],generated_tokens=len(r['generated_ids'])))
        assert session.status()==before and session.bank.selector_digest()==fingerprint
        session.close();assert not session.bank.banks and not session.bank.scorers
    dependencies=[]
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            path=Path(path).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT.resolve()):
                rel=path.relative_to(ROOT.resolve()).as_posix()
                if rel.startswith(('experiments/','scripts/','publish/','repo/')):dependencies.append(rel)
    with out.open('x',encoding='utf-8') as f:json.dump(dict(status='passed',real_RGB_pairs=9,
        loaded_source_files=sorted(set(dependencies)),
        tiny_visual_calls=45,tiny_QA_calls=5,rows=rows,eight_pair_coverage=True,bank_selector_unchanged=True,
        original_R2_fixed_paths_exact=True,
        branch_released=True,close_releases_banks_and_scorers=True,GPU=0,pretrained_loads=0,quality_claim=False),f,indent=2)
    print('Nine RGB pairs / five bounded banks / five QA / close passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
