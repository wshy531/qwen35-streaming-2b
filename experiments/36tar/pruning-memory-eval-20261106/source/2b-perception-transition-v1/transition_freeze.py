"""Freeze the unchanged shared mechanics plus this experiment. No submission."""
import argparse
from pathlib import Path
import sys
from transition_common import ROOT, HERE, PREP, INPUTS, read, save, sha


def old_verifier():
    sys.path.insert(0,str(ROOT/'experiments/2b-perception-interference-v1'))
    from freeze_execution import verify_lock
    return verify_lock


def verify(path):
    lock=read(path)
    if lock['schema']!='perception-transition-execution-v1':
        raise ValueError('fixed transition lock required')
    old_verifier()(ROOT/lock['shared_lock_path'])
    for name,digest in lock['sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError('changed source/input/proof: '+name)
    return lock


def main(output,cpu,coordinator):
    paths=[Path(p).resolve() for p in (output,cpu,coordinator)]
    output,cpu,coordinator=paths
    if not all(p.is_relative_to((ROOT/'runs').resolve()) for p in paths):
        raise ValueError('owned result paths required')
    old_path=ROOT/'runs/perception_interference_prepare_20261027_01/execution_lock.json'
    old_verifier()(old_path)
    proof=read(cpu)
    if proof.get('historical_text_inputs_reproduced')!=1020 or proof.get('planned_transition_answers')!=270:
        raise ValueError('complete canonical input CPU proof required')
    coord=read(coordinator)
    if coord.get('status')!='passed_CPU_transition_coordinator':
        raise ValueError('new coordinator CPU proof required')
    for mapping in (proof['source_sha256'],coord['source_sha256']):
        for name,digest in mapping.items():
            if sha(ROOT/name)!=digest: raise ValueError('CPU-tested source changed: '+name)
    files={p for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')}
    files.update(p for p in INPUTS.iterdir() if p.is_file())
    files.update([cpu,coordinator,old_path])
    files.update(PREP/name for name in ('canonical_input_reference_LOCAL_ONLY.json',
        'canonical_input_reference_receipt.json','canonical_requests_LOCAL_ONLY.json','assistant_visual_review_LOCAL_ONLY.json',
        'review_card_receipt_LOCAL_ONLY.json','metadata_amendment_v1.json'))
    for case in read(PREP/'assistant_visual_review_LOCAL_ONLY.json')['cases']:
        files.update(ROOT/image['relative_path'] for image in case['evidence'])
    result=dict(schema='perception-transition-execution-v1',shared_lock_path=old_path.relative_to(ROOT).as_posix(),
        sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        budget=dict(GPUs=1,CPUs=4,RAM_GiB=24,minutes=20,formal_answers=300,warmup_answers=2,
            max_generated_tokens=2416,visual_calls=804,pretrained_loads=1,training_steps=0,
            downloads=0,requeue=False,automatic_retry=False),
        reference_gate='30/30 before transition phase',GPU_authorization_not_created=True)
    save(output,result)
    verify(output)
    print(dict(status='execution_frozen_not_authorized',files=len(files),budget=result['budget']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--coordinator',type=Path,required=True)
    args=parser.parse_args()
    main(args.output,args.cpu,args.coordinator)
