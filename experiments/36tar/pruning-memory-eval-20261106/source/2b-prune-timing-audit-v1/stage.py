"""Only new audit files, no media or credentials; does not submit jobs."""
import base64,subprocess,shlex
from audit_common import *
files=[p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.sh','.md')]
files += [p for p in PREP.iterdir() if p.suffix=='.json' and p.name not in ('GPU_authorization.json','submission_attempt.json','submission_receipt.json')]
payload={p.relative_to(ROOT).as_posix():base64.b64encode(p.read_bytes()).decode() for p in files}
remote='''import base64,json,hashlib,sys
from pathlib import Path
r=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert r.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
data=json.load(sys.stdin)
for n,b in data.items():
 p=r/n;assert p.resolve().is_relative_to(r.resolve())
 assert n.startswith(('experiments/2b-prune-timing-audit-v1/','runs/prune_timing_prepare_20261103_01/'))
 raw=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==raw,n
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
print(json.dumps(dict(files=len(data),bytes=sum(len(base64.b64decode(x)) for x in data.values()))))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','ChaoSuan','python -c '+shlex.quote(remote)],
    input=json.dumps(payload),text=True,capture_output=True)
print(r.stdout);print(r.stderr);assert r.returncode==0
