"""Fetch only bounded artifacts of approved job217135; no remote writes/GPU."""
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/first_2b_gpu_217135_20261020_01'
REMOTE=r'''
import getpass,json,re,sys,tarfile
from pathlib import Path
p=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert getpass.getuser()=='u2025010462' and Path.cwd().resolve()==p.resolve()
assert p.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
r=p/'runs/first-2b-gpu-217135';files={}
for stage,n in [('original',16),('stream',8)]:
 s=json.loads((r/stage/'summary.json').read_text());assert s['status']=='complete' and s['stage']=='gpu' and s['pretrained_weights_loaded'] and len(s['records'])==n
 for name in ['summary.json','resource_guard.json']:files[stage+'/'+name]=r/stage/name
 for row in s['records']:
  name=row['logits_file'];assert re.fullmatch(r'[0-3]-(2|4)-(native|matched|stream)\.npy',name)
  files[stage+'/'+name]=r/stage/name
for name in ['comparison/comparison.json','original.log','stream.log','comparison.log','job_id.txt','hostname.txt']:files[name]=r/name
for ext in ['out','err']:files['slurm.'+ext]=p/('logs/first-2b-217135.'+ext)
code=p/'repo/experiments/36tar/2b-memory-eval'
for name in ['first_2b_baseline.py','analyze_first_baseline.py','run_first_2b_baseline.sh']:files['source_snapshot/'+name]=code/name
assert all(q.is_file() and q.resolve().is_relative_to(p.resolve()) for q in files.values())
assert sum(q.stat().st_size for q in files.values())<64*2**20
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,q in files.items():archive.add(q,arcname=name,recursive=False)
'''

assert not OUT.exists(),'new artifact directory required'
submission=json.loads((ROOT/'runs/first_2b_submission_20261020_01/submission_result.json').read_text())
assert submission['job_id']=='217135' and submission['exit_code']==0
project='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
cmd=['C:/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i',str(Path.home()/'.ssh/qwen35_cluster_temp'),'ChaoSuan',
    'cd '+project+' && PYTHONDONTWRITEBYTECODE=1 '+project+'/envs/qwen35-2b-fp32/bin/python -']
proc=subprocess.run(cmd,input=REMOTE.encode(),capture_output=True,timeout=90)
if proc.returncode:raise RuntimeError('Result not ready or SSH failed: '+proc.stderr.decode(errors='replace')[-1200:])
assert len(proc.stdout)<=65*2**20
prepared={};total=0
with tarfile.open(fileobj=io.BytesIO(proc.stdout),mode='r:') as archive:
    for member in archive:
        relative=Path(member.name)
        assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts and ':' not in member.name and '\\' not in member.name
        assert member.name not in prepared and member.size<=2*2**20
        content=archive.extractfile(member).read();assert len(content)==member.size
        total+=len(content);assert total<64*2**20;prepared[member.name]=content
assert len(prepared)<=64
OUT.mkdir()
for name,content in prepared.items():
    dest=OUT/name;assert dest.resolve().is_relative_to(OUT.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
receipt=dict(status='fetched_exact_approved_job_artifacts',job_id='217135',remote_project=project,files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in prepared.items()},
    count=len(prepared),total_bytes=total,transport_bytes=len(proc.stdout),no_models_or_media_fetched=True,new_GPU_commands=0)
(OUT/'fetch_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
