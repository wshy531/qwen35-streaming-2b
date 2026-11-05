import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/recent_detail_prepare_20261103_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((R/'runs/midprune_memory_prepare_20261102_01/execution_lock.json').read_text(encoding='utf-8'))
for n,h in lock.items():
    p=R/n
    if not p.exists() and n.startswith('repo/'):p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
for p in [*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.sh'),*Path(__file__).parent.glob('*.md'),
          *[P/n for n in ('questions.json','sources.json','labels_LOCAL_ONLY.json','protocol.json')]]:
    lock[p.relative_to(R).as_posix()]=sha(p)
(P/'execution_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(lock),sha256=sha(P/'execution_lock.json'))))
