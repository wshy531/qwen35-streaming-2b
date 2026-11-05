"""Only this new experiment and its small preparation artifacts; no job submit."""
from pathlib import Path
import json,base64,subprocess,shlex
r=Path(__file__).resolve().parents[2]
dirs=['experiments/2b-memory-followup-v1','runs/memory_followup_prepare_20261103_01']
files=[p for n in dirs for p in (r/n).iterdir() if p.suffix in ('.py','.sh','.json','.md')]
payload={p.relative_to(r).as_posix():base64.b64encode(p.read_bytes()).decode() for p in files}
code="""import sys,json,base64
from pathlib import Path
r=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert r.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
d=json.load(sys.stdin)
for n,b in d.items():
 assert n.startswith(('experiments/2b-memory-followup-v1/','runs/memory_followup_prepare_20261103_01/'))
 p=r/n;assert p.resolve().is_relative_to(r.resolve());raw=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==raw,n
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
print(json.dumps(dict(files=len(d),bytes=sum(len(base64.b64decode(b)) for b in d.values()))))
"""
z=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','ChaoSuan','python -c '+shlex.quote(code)],input=json.dumps(payload),text=True,capture_output=True)
print(z.stdout,z.stderr);assert z.returncode==0
