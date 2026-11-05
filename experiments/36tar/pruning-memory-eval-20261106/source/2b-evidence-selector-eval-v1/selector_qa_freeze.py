"""Immutable evaluation closure, trained weights and prepared inputs."""
import argparse,sys
from pathlib import Path
from selector_qa_common import ROOT,PREP,HEADS,read,save,sha

def inherited_verify():
    sys.path.insert(0,str(ROOT/'experiments/2b-evidence-selector-train-v1'))
    from selector_freeze import verify as old_verify
    old_verify()

def verify():
    inherited_verify();lock=read(PREP/'execution_lock.json')
    for p,h in lock['sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen source/input changed: '+p)
    return lock

def main():
    inherited_verify()
    files=[p for p in Path(__file__).resolve().parent.iterdir() if p.suffix in ('.py','.md','.sh')]
    files += [PREP/n for n in ('requests.json','labels_LOCAL_ONLY.json','plan.json','protocol.json',
        'references_LOCAL_ONLY.json','head_lock.json','local_CPU_preflight.json','cluster_CPU_preflight.json',
        'cluster_media_audit.json','two_site_comparison.json','local_scoring_smoke.json','cluster_scoring_smoke.json')]
    files += [HEADS/(n+s) for n in ('content_visible','history_blind') for s in ('.pt','_predictions.json')]
    files += [ROOT/'runs/selector-features-219523/features.npz']
    save(PREP/'execution_lock.json',dict(sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        budget=read(PREP/'protocol.json'),inherited_runtime='selector-train-v1 frozen closure',
        authorization_not_generated=True))
    verify();print('Execution lock verified:',len(files),'evaluation files plus inherited runtime')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:print('Verified',len(verify()['sha256']),'evaluation files plus inherited runtime')
    else:main()
