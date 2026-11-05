"""New feature-stage lock plus unchanged completed evidence-diagnostic runtime."""
from pathlib import Path
import sys,json
from prepare_data import ROOT,PREP,read,save,sha
HERE=Path(__file__).resolve().parent

def verify():
    lock=read(PREP/'execution_lock.json')
    sys.path.insert(0,str(ROOT/'experiments/2b-evidence-selection-v1'))
    from execution_freeze import verify as base_verify
    base_verify()
    for p,h in lock['sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('selector input/source changed: '+p)
    return lock

def main():
    files=[p for p in HERE.iterdir() if p.suffix in ('.py','.sh','.md')]
    files += [PREP/n for n in ('short_requests.json','short_sources.json','supervision_LOCAL_ONLY.json',
        'long_requests.json','long_sources.json','long_supervision_LOCAL_ONLY.json','protocol.json',
        'local_CPU_preflight_final.json','cluster_CPU_preflight.json','cluster_asset_audit.json')]
    files += [ROOT/'runs/router-head-219466/features.npz']
    lock=dict(sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        budget=dict(GPUs=1,CPUs=4,RAM_GiB=24,minutes=45,vision_calls=7640,text_prefills=104,
            pretrained_loads=1,generated_tokens=0,base_updates=0,API_calls=0,downloads=0,
            feature_file_MiB=64,total_result_MiB=96,log_MiB=64,automatic_retry=False,requeue=False),
        head_training_after_export_on_CPU=True,GPU_authorization_not_created=True)
    save(PREP/'execution_lock.json',lock);verify();print(dict(files=len(files),budget=lock['budget']))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:print(verify()['budget'])
    else:main()
