"""Test new stage gating and causal checkpoint orchestration without weights."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
import torch
from transition_common import ROOT, HERE, PREP, INPUTS, read, save, sha
import transition_worker as worker


def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':
        raise RuntimeError('CPU only')
    from test_transition_scoring import main as scoring_check
    scoring=scoring_check()
    expected=read(worker.OLD/'old_reference_predictions_LOCAL_ONLY.json')
    canonical=read(PREP/'canonical_input_reference_LOCAL_ONLY.json')
    canonical_keys={(r['id'],r['arm'],r['shift']) for r in canonical['rows']}
    old_pairs=read(worker.OLD/'inputs/manifest_LOCAL_ONLY.json')['pairs']
    for pair in old_pairs:
        for arm in ('recent','full'):
            for shift in range(5):
                assert (pair['old_question_id'],arm,shift) in canonical_keys
    raw=[dict(task_id=r['task_id'],arm=r['arm'],shift=r['shift'],
              text='unknown' if r['expected_content_choice_original'] is None else
              chr(65+(r['expected_content_choice_original']-r['shift'])%5)) for r in expected]
    assert worker.reference_gate(raw,expected)['passed']
    raw[0]['text']='unknown' if raw[0]['text']!='unknown' else 'A'
    assert not worker.reference_gate(raw,expected)['passed']
    calls=[]
    class FakeBank:
        def __init__(self,*args): self.seen=[]
        def status(self): return dict(tokens=864,recent_tokens=432)
    def pairs(path,trace):
        for index in range(len(trace)//2):
            yield SimpleNamespace(timestamp=index*2),SimpleNamespace(timestamp=index*2+1)
    def project(env,model,builder,a,b):
        calls.append(b.timestamp)
        return torch.zeros(108,2),None
    def append(bank,value,index,a,b,previous):
        bank.seen.append(b.timestamp)
        return value.mean(0)
    with patch.object(worker,'shared',return_value=(pairs,project,append,None,None,None,FakeBank)):
        observed=[]
        for when,bank,count in worker.checkpoints(None,None,None,Path('synthetic'),list(range(32)),[15,31],lambda:None):
            assert max(bank.seen)==when and count==(when+1)//2
            observed.append(when)
        assert observed==[15,31] and len(calls)==16
        try:
            list(worker.checkpoints(None,None,None,Path('synthetic'),list(range(32)),[30],lambda:None))
        except ValueError: pass
        else: raise AssertionError('unreachable query accepted')
    forbidden=ROOT/'runs/CPU_must_not_start_GPU_transition'
    assert not forbidden.exists()
    with patch.dict(os.environ,{'SLURM_JOB_ID':'','SLURM_CPUS_PER_TASK':'4'}):
        try: worker.main(forbidden,Path('nonexistent-lock'),Path('nonexistent-media'))
        except RuntimeError as error: assert 'Slurm' in str(error)
        else: raise AssertionError('non-Slurm GPU worker accepted')
    assert not forbidden.exists()
    review=read(PREP/'assistant_visual_review_LOCAL_ONLY.json')
    assert review['status']=='completed_assistant_development_review' and review['new_model_answers']==0
    assert review['cards_actually_viewed']==12 and len(review['cases'])==6
    for case in review['cases']:
        assert case['current_early_and_late_sufficient'] and not case['independent_review']
        for image in case['evidence']: assert sha(ROOT/image['relative_path'])==image['sha256']
    sources=read(INPUTS/'sources_LOCAL_ONLY.json')
    target=sum(len(s['sampled_trace'])//2 for s in sources)
    old=sum(len(s['sampled_trace'])//2 for s in read(worker.OLD/'cluster_media_CPU.json')['sources'])
    assert target==429 and old==355 and old+target+len(sources)*4==804
    result=dict(status='passed_CPU_transition_coordinator',scoring_check=scoring,
        old_reference_match_and_mismatch_checked=True,causal_checkpoints_checked=True,
        all_30_reference_canonical_keys_checked=True,
        missing_query_rejected=True,non_Slurm_preload_rejected=True,
        reference_pairs=old,target_pairs=target,donor_pairs=20,total_visual_calls=804,
        formal_answers=300,warmup_answers=2,max_generated_tokens=2416,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')},
        source_order=[s['name'] for s in sources],
        donor_mapping={s['name']:sources[(i+1)%len(sources)]['name'] for i,s in enumerate(sources)},
        no_new_shared_math=True,real_weight_loads=0,GPU_jobs=0)
    save(output,result)
    print({k:v for k,v in result.items() if k not in ('source_sha256','source_order','donor_mapping','scoring_check')})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
