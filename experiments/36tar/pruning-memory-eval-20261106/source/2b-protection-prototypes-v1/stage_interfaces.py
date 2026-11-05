"""New project-scoped source files only; no jobs or model/media transfers."""
import base64,json,subprocess,shlex
from pathlib import Path
R=Path(__file__).resolve().parents[2]
folders=['experiments/2b-protection-prototypes-v1','experiments/2b-anchor-merge-v1','experiments/2b-isolated-recent-v1']
files=[p for f in folders for p in (R/f).iterdir() if p.suffix in ('.py','.md')]
payload={p.relative_to(R).as_posix():base64.b64encode(p.read_bytes()).decode() for p in files}
code='''import sys,json,base64,hashlib
from pathlib import Path
r=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert r.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
d=json.load(sys.stdin)
for n,b in d.items():
 p=r/n;assert p.resolve().is_relative_to(r.resolve())
 assert n.startswith(('experiments/2b-protection-prototypes-v1/','experiments/2b-anchor-merge-v1/','experiments/2b-isolated-recent-v1/'))
 raw=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==raw,n
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
print(json.dumps(dict(files=len(d),bytes=sum(len(base64.b64decode(x)) for x in d.values()))))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','ChaoSuan','python -c '+shlex.quote(code)],input=json.dumps(payload),text=True,capture_output=True)
print(r.stdout);print(r.stderr);assert r.returncode==0
