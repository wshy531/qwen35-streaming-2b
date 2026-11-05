import json,base64,subprocess,shlex
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/protection_channels_prepare_20261103_01'
files=[p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.sh','.md')]
files += [p for p in P.iterdir() if p.suffix=='.json' and p.name not in ('CPU_cluster.json','submission.json','submit_intent.json')]
files += [R/'experiments/2b-protection-prototypes-v1/channel_intervention.py']
payload={p.relative_to(R).as_posix():base64.b64encode(p.read_bytes()).decode() for p in files}
code='''import sys,json,base64,hashlib
from pathlib import Path
r=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert r.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
d=json.load(sys.stdin)
for n,b in d.items():
 assert n.startswith(('experiments/2b-protection-channels-v1/','runs/protection_channels_prepare_20261103_01/')) or n=='experiments/2b-protection-prototypes-v1/channel_intervention.py'
 p=r/n;assert p.resolve().is_relative_to(r.resolve());raw=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==raw,n
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
print(json.dumps(dict(files=len(d),bytes=sum(len(base64.b64decode(x)) for x in d.values()))))
'''
z=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','ChaoSuan','python -c '+shlex.quote(code)],input=json.dumps(payload),text=True,capture_output=True)
print(z.stdout);print(z.stderr);assert z.returncode==0
