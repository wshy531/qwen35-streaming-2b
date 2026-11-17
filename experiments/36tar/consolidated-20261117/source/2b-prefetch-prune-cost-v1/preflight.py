"""Run the actual cost worker with random tiny weights and real synthetic decoding."""
import argparse
import os
import sys
from common import ROOT, PREP, CASES, MATCH, read, write, sha, worker, run_case

def main(side):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=worker.scale.s.runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(1109)
    proc=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](proc);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder=worker.scale.s.make_builder(env,proc)
    versions={n:p._version for n,p in model.named_parameters()}
    q=dict(read(PREP/'panel.json')[0],media_path=(PREP/'fixture.mp4').relative_to(ROOT).as_posix(),query_time_s=3.0)
    original=worker.timed_generate
    def short(*args,**kwargs):
        return original(*args,**kwargs,limit=2)
    worker.timed_generate=short
    rows={};serial={}
    try:
        with t.no_grad(),env['fp32_reference']():
            for case in CASES:
                rows[case]=run_case(env,model,builder,q,case,lambda:None)
            from common import V0Pairs
            worker.V0Pairs=V0Pairs
            for arm in ('full','fixed12','heuristic11'):
                serial[arm]=worker.run_source(env,model,builder,q,arm,lambda:None)
        for case,(_,arm) in CASES.items():
            assert all(rows[case][k]==serial[arm][k] for k in MATCH),case
            times=rows[case]['timing']
            assert all(times[k]>=0 for k in ('context_seconds','prefill_seconds','generation_seconds','QA_total_seconds'))
        assert versions=={n:p._version for n,p in model.named_parameters()}
        assert all(p.grad is None for p in model.parameters())
    finally:worker.timed_generate=original
    modules={}
    for m in list(sys.modules.values()):
        file=getattr(m,'__file__',None)
        if file:
            p=__import__('pathlib').Path(file).resolve()
            if p.is_relative_to(ROOT.resolve()) and p.suffix=='.py' and ('experiments' in p.parts or 'streaming' in p.parts):
                modules[p.relative_to(ROOT).as_posix()]=sha(p)
    compact={case:{k:r[k] for k in MATCH} for case,r in rows.items()}
    result=dict(status='passed_random_weights_only',side=side,answers=7,pretrained_loads=0,GPU=0,
                endpoint_match=True,model_unchanged=True,outputs=compact,loaded_project_source_SHA=modules)
    write(PREP/f'CPU_{side}.json',result)
    print('CPU_PASS',side,len(modules))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--side',required=True,choices=['local','remote']);main(p.parse_args().side)
