import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/hard_mask_main_prepare_20261102_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((R/'runs/hard_mask_prepare_20261102_01/execution_lock.json').read_text())
for n,h in old.items():
    p=R/n
    if not p.exists() and n.startswith('repo/'):p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))
files += [P/n for n in ('protocol.json','baseline_answers.json','reference_subset.json','reference_questions.json','training_probes.json')]
files += [R/'runs/hard-mask-profile-220060/completed.json']
for p in files:old[p.relative_to(R).as_posix()]=sha(p)
(P/'execution_lock.json').write_text(json.dumps(old,indent=2)+'\n')
print(json.dumps(dict(files=len(old),sha256=sha(P/'execution_lock.json'))))
