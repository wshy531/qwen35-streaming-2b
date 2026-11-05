"""Actual worker packing and complete scheduled context layouts; random tiny weights."""
import os,sys,argparse,json
from pathlib import Path
from common import ROOT,PREP,read,save,pack,Session,MODES,runtime,make_builder,prompt
def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261101)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor,(288,384));qs=read(PREP/'requests.json')
    from quality_banks import selected_node
    from resolution_runtime import context
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    exp=read(PREP/'expected_prefixes.json');layouts=[]
    with t.inference_mode():
        for q in qs:
            end=exp[q['question_id']]['pairs'];ids=list(range(max(0,end-8),end))
            for mode in MODES:
                indices=t.arange(108) if mode=='full' else t.arange(1,108,2)
                nodes=[selected_node(t.zeros(len(indices),32),indices,i,4.*i,4.*i+2) for i in ids]
                emb,pos,text,meta=context(model,builder,nodes,prompt(q),visual_limit=864,max_new_tokens=16)
                assert emb.shape[1]==pos.shape[-1]==text.shape[1] and len(nodes)<=8
                layouts.append(dict(question_id=q['question_id'],mode=mode,visual_tokens=meta['visual_tokens'],input_tokens=meta['input_tokens']))
        from types import SimpleNamespace
        import numpy as np
        session=Session(env,model,builder)
        a=SimpleNamespace(frame_id=0,timestamp=0.,rgb=np.zeros((288,384,3),np.uint8))
        b=SimpleNamespace(frame_id=1,timestamp=2.,rgb=np.ones((288,384,3),np.uint8))
        session.append_pair(a,b)
        for mode in MODES:
            r=session.ask(mode,'What is visible? A. Cup. B. Book.',lambda item:None,max_new_tokens=2)
            row=pack(qs[0],dict(pairs=1),r);assert json.loads(json.dumps(row))==row and row['arm']==mode
        session.close()
    import worker
    old=os.environ.pop('SLURM_JOB_ID',None)
    try:
        try:worker.main(0,ROOT/'runs/forbidden_content_main',0.)
        except RuntimeError as e:assert 'Slurm' in str(e)
        else:raise AssertionError('missing allocation accepted')
    finally:
        if old:os.environ['SLURM_JOB_ID']=old
    deps=[]
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT.resolve()):
                rel=p.relative_to(ROOT.resolve()).as_posix()
                if rel.startswith(('experiments/','scripts/','repo/','publish/')):deps.append(rel)
    save(out,dict(status='passed',scheduled_layouts=len(layouts),tiny_actual_record_roundtrips=5,
        Slurm_missing_rejected=True,pretrained_loads=0,GPU=0,loaded_source_files=sorted(set(deps)),
        layout_summary=layouts,quality_claim=False))
    print('Passed 2390 layouts / five actual QA-to-JSON / allocation guard')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
