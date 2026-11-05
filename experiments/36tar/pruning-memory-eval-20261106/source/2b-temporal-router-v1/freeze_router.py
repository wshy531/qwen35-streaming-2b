"""Freeze CPU-tested candidate, cached evaluation evidence and isolated runtime."""
import argparse
from pathlib import Path
import sys
import importlib.metadata
from route_common import ROOT,HERE,INPUTS,read,save,sha


def shared_verify(path):
    sys.path.insert(0,str(ROOT/'experiments/2b-perception-transition-v1'))
    from transition_freeze import verify
    return verify(path)


def verify(path):
    lock=read(path)
    if lock['schema']!='temporal-router-execution-v1':raise ValueError('frozen execution lock required')
    shared_verify(ROOT/lock['shared_lock_path'])
    for name,digest in lock['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('changed code/input/proof: '+name)
    for name,value in lock['package_versions'].items():
        if importlib.metadata.version(name)!=value:raise ValueError('package version changed: '+name)
    for name,digest in lock['environment_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('isolated environment changed: '+name)
    return lock


def main(output,cpu,tiny):
    output,cpu,tiny=[Path(p).resolve() for p in (output,cpu,tiny)]
    if not all(p.is_relative_to((ROOT/'runs').resolve()) for p in (output,cpu,tiny)):
        raise ValueError('owned output/proof paths required')
    proof=read(cpu);smoke=read(tiny)
    if proof['status']!='passed_CPU_temporal_router' or proof['prompt_layouts']!=1017:
        raise ValueError('complete CPU prompt check required')
    if smoke['status']!='passed_CPU_temporal_router' or smoke['tiny_shared_prefills']!=10:
        raise ValueError('shared tiny hybrid prefill check required')
    for name,digest in proof['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('CPU-tested source changed: '+name)
    # The tiny forward depends on these functions; final CLI/layout proof binds
    # every source, including subsequently finalized scheduler/cache settings.
    shared_names=('route_runtime.py','temporal_policy.py','preflight_router.py')
    for name,digest in smoke['source_sha256'].items():
        if Path(name).name in shared_names and sha(ROOT/name)!=digest:
            raise ValueError('tiny-tested shared function changed: '+name)
    files={p for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')}
    files.update(p for p in INPUTS.iterdir() if p.is_file());files.update([cpu,tiny])
    shared=ROOT/'runs/perception_transition_prepare_20261027_01/execution_lock.json'
    shared_verify(shared)
    files.add(shared)
    base=Path(sys.prefix).resolve()
    if base.name!='temporal-router-v1' or not base.is_relative_to((ROOT/'envs').resolve()):
        raise ValueError('own isolated linked-dependency environment required')
    environment=[base/'pyvenv.cfg',base/'lib/python3.11/site-packages/project_runtime_dependencies.pth']
    result=dict(schema='temporal-router-execution-v1',shared_lock_path=shared.relative_to(ROOT).as_posix(),
        sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        environment_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in environment},
        package_versions={p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','psutil')},
        budget=dict(GPUs=1,CPUs=4,RAM_GiB=24,minutes=15,pretrained_loads=1,
            formal_classifier_forwards_max=1017,warmup_classifier_forwards=9,total_classifier_forwards_max=1026,
            generated_tokens=0,new_QA_answers=0,visual_forwards=0,training_steps=0,downloads=0,
            automatic_retry=False,requeue=False),
        GPU_authorization_not_created=True,all_branch_answers_cached=True)
    save(output,result);verify(output)
    print(dict(status='frozen_not_authorized',files=len(files),budget=result['budget']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',type=Path,required=True);p.add_argument('--tiny',type=Path,required=True)
    a=p.parse_args();main(a.output,a.cpu,a.tiny)
