import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((R/'runs/learned_prune_main_prepare_20261101_01/execution_lock.json').read_text())
for n,h in lock.items():
    p=R/n
    if not p.exists() and n.startswith('repo/'):p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))
files+=[P/n for n in ('profile.json','panel.json','controls.json','labels_nextqa.json','selection_summary.json')]
for p in files:lock[p.relative_to(R).as_posix()]=sha(p)
(P/'execution_lock.json').write_text(json.dumps(lock,indent=2)+'\n');print(len(lock))
