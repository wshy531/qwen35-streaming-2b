"""Prepare a bounded, project-only transfer and verify remote bytes. No submission."""
import base64,hashlib,json,shlex,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/recent_detail_prepare_20261103_01'
prior=R/'runs/midprune_memory_prepare_20261102_01'
P.joinpath('labels_LOCAL_ONLY.json').write_bytes((prior/'labels_LOCAL_ONLY.json').read_bytes())
spec=json.loads((prior/'protocol.json').read_text(encoding='utf-8'))
spec.update(stage='Borrowed latest-pair detail baseline, familiar development pilot',arms=['uniform864','uniform810','protect864'],
    answers=54,warmup_answers=3,max_generated_tokens=6915,visual_calls=429,model_loads=1,
    memory='Latest8 pair coverage shared; protect vs810 identical history, full latest pair; no main KV or full history archive',
    budget=dict(partition='g078t',comment='gx_llm',GPU=1,cpus=4,memory_GiB=24,minutes=20,output_MiB=32))
for k in ('checkpoint','checkpoint_sha256','views','policies','matched_coverage_controls'):spec.pop(k,None)
P.joinpath('protocol.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
subprocess.run([str(R/'envs/recent-detail-baseline-v1/Scripts/python.exe'),str(Path(__file__).with_name('freeze.py'))],check=True)
files=[p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.sh','.md')]
files += [p for p in P.iterdir() if p.suffix=='.json' and p.name not in ('GPU_authorization.json','submission_attempt.json','submission_receipt.json')]
payload={p.relative_to(R).as_posix():base64.b64encode(p.read_bytes()).decode() for p in files}
remote="""import base64,hashlib,json,sys
from pathlib import Path
r=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert r.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
data=json.load(sys.stdin)
old=json.loads((r/'runs/recent_detail_prepare_20261103_01/execution_lock.json').read_text()) if (r/'runs/recent_detail_prepare_20261103_01/execution_lock.json').exists() else {}
for n,b in data.items():
 p=r/n;assert p.resolve().is_relative_to(r.resolve())
 assert n.startswith(('experiments/2b-recent-detail-baseline-v1/','runs/recent_detail_prepare_20261103_01/'))
 raw=base64.b64decode(b)
 if p.exists() and p.read_bytes()!=raw:
  if n.endswith('/execution_lock.json'):assert old
  else:assert old.get(n)==hashlib.sha256(p.read_bytes()).hexdigest(),n
  backup=r/'runs/recent_detail_prepare_20261103_01/entry_import_failure'/n
  assert not backup.exists();backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(p.read_bytes())
  p.write_bytes(raw)
 elif p.exists():assert p.read_bytes()==raw,n
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
print(json.dumps(dict(verified_files=len(data),bytes=sum(len(base64.b64decode(x)) for x in data.values()))))
"""
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','ChaoSuan','python -c '+shlex.quote(remote)],
 input=json.dumps(payload),text=True,capture_output=True)
print(result.stdout);print(result.stderr);assert result.returncode==0
