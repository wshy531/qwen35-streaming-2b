"""Freeze exact runtime inherited from the completed reference and new code/data."""
import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from prepare import ROOT,PREP,read,save,sha
HERE=Path(__file__).resolve().parent

def runtime_path(name):
    if name.startswith('repo/') and not (ROOT/'repo').exists():return ROOT/'publish/qwen35-streaming-2b'/name[5:]
    return ROOT/name

def verify():
    lock=read(PREP/'execution_lock.json')
    for k,v in lock['runtime_sha256'].items():
        if sha(runtime_path(k))!=v:raise ValueError('runtime SHA mismatch: '+k)
    for k,v in lock['sha256'].items():
        if sha(ROOT/k)!=v:raise ValueError('source/input SHA mismatch: '+k)
    expected=lock['local_cpu_package_versions'] if sys.platform=='win32' else lock['package_versions']
    for k,v in expected.items():
        if version(k)!=v:raise ValueError('dependency version changed: '+k)
    for k,v in lock['model_metadata_sha256'].items():
        if sha(ROOT/'models/Qwen3.5-2B'/k)!=v:raise ValueError('model/processor metadata changed')
    return lock

def main():
    base=read(ROOT/'runs/perception_interference_prepare_20261027_01/execution_lock.json')
    files=[p for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')]
    files += [PREP/n for n in ('runtime_plan.json','protocol.json','decision_gate_before_GPU.json',
        'inputs/requests.json','audit/evidence_bindings_LOCAL_ONLY.json','audit/labels_LOCAL_ONLY.json',
        'local_real_layout_preflight.json','cluster_real_layout_preflight.json','cluster_media_header_audit.json',
        'two_site_CPU_identity.json','worker_CPU_smoke.json','cluster_worker_CPU_smoke.json','exact_workload.json','cluster_environment.json')]
    # The inherited closure swept every *.py under the source tree, including
    # tests not imported by this runtime. Keep production digests strict.
    runtime={k:v for k,v in base['runtime_sha256'].items() if '/tests/' not in k}
    for k,v in runtime.items():
        if sha(runtime_path(k))!=v:raise ValueError('production runtime drift: '+k)
    cluster=read(PREP/'cluster_environment.json')
    if cluster['packages']!=base['package_versions']:raise ValueError('GPU environment differs from completed reference')
    lock=dict(runtime_sha256=runtime,package_versions=base['package_versions'],
        local_cpu_package_versions={k:version(k) for k in base['package_versions']},
        excluded_unexecuted_test_files={k:dict(inherited_sha=v,current_local_sha=sha(runtime_path(k)))
            for k,v in base['runtime_sha256'].items() if '/tests/' in k},
        model_metadata_sha256=base['model_metadata_sha256'],sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        budget=dict(GPUs=1,CPUs=4,RAM_GiB=24,minutes=45,visual_forwards=2796,
            formal_slots=1600,actual_model_answers_max=1105,warmup_answers=2,generation_tokens_max=8856,
            pretrained_loads=1,training_steps=0,API_requests=0,downloads=0,requeue=False,automatic_retry=False,
            result_MiB=64,log_MiB=16),approval_not_created_by_this_script=True)
    save(PREP/'execution_lock.json',lock);verify();print(dict(runtime_files=len(lock['runtime_sha256']),new_files=len(files),budget=lock['budget']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:print(verify()['budget'])
    else:main()
