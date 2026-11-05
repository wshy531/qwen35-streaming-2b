"""Lock inherited code unchanged, then this independent pilot's inputs and implementation."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/midprune_memory_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(n):
    p=R/n
    if not p.exists() and n.startswith('repo/'):p=R/'publish/qwen35-streaming-2b'/n[5:]
    return p
lock=read(R/'runs/prune_checkpoints_prepare_20261102_01/execution_lock.json')
for n,h in lock.items():assert sha(resolve(n))==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+list(Path(__file__).parent.glob('*.md'))
files+=list((R/'experiments/2b-resolution-cost-v1').glob('*.py'))
files+=[R/'experiments/2b-perception-interference-v1/audit_media.py']
files+=[P/n for n in ('questions.json','sources.json','labels_LOCAL_ONLY.json','protocol.json','provenance.json')]
for p in files:lock[p.relative_to(R).as_posix()]=sha(p)
(P/'execution_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(lock),sha256=sha(P/'execution_lock.json'))))
