"""Freeze the prepared run and inherited source dependency chain."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'runs/prune_memory_completion_prepare_20261105_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    inherited=read(ROOT/'runs/memory_prune_panel150_prepare_20261103_01/execution_lock.json')
    lock={}
    for rel,h in inherited.items():
        path=ROOT/rel
        if rel.startswith('repo/') and not path.exists():path=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        if not path.exists():
            # Files already present only on cluster retain the frozen digest.
            lock[rel]=h;continue
        assert sha(path)==h,rel
        lock[rel]=h
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))
    paths+=[PREP/n for n in ('plan.json','references.json','stream_questions.json','stream_sources.json','paired_questions.json','paired_sources.json','reference_receipt.json')]
    paths+=[ROOT/'runs/repair-guard-220499/final.pt',ROOT/'runs/training-order-main-220483/initial.pt',
            ROOT/'runs/hard-mask-main-220062/mask_ce_step1237.pt']
    for p in paths:lock[p.relative_to(ROOT).as_posix()]=sha(p)
    (PREP/'execution_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(lock),sha256=sha(PREP/'execution_lock.json'))))

if __name__=='__main__':main()
