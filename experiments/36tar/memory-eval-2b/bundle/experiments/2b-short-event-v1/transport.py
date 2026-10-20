"""Transfer only owned small files and run bounded CPU preflight. No GPU submission."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import bridge

HERE,ROOT,BASE=bridge.HERE,bridge.ROOT,bridge.BASE
from cluster_io import ssh, command, PROJECT, PY
LOCAL=ROOT/'runs/2b_short_event_prepare_20261020_01'
REMOTE_INPUT='runs/2b-short-event-inputs-20261020-01'
REMOTE_CODE='experiments/2b-short-event-v1'
ALLOWED_INPUTS=['requests.json','diagnostic_requests.json','protocol.json','source_lock.json',
                'dependency_lock.json','baseline_timelines.json']

def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def deploy():
    archive=LOCAL/'source_transfer.tar'
    expected={}
    with tarfile.open(archive,'x') as tar:
        for folder,rel,paths in [(HERE,REMOTE_CODE,[p.name for p in HERE.iterdir() if p.suffix in ['.py','.sh','.md']]),
                                 (LOCAL,REMOTE_INPUT,ALLOWED_INPUTS)]:
            for name in paths:
                p=folder/name
                assert p.stat().st_size<2*2**20
                dest=rel+'/'+name
                tar.add(p,arcname=dest,recursive=False)
                expected[dest]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if archive.stat().st_size>5*2**20:raise ValueError('small text-only transfer bound')
    code="""import hashlib,json,pathlib,tarfile,sys
p=pathlib.Path.cwd().resolve();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
if not p.is_relative_to(allowed):raise ValueError('outside authorized root')
items={};total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  rel=pathlib.PurePosixPath(m.name)
  if not m.isfile() or '..' in rel.parts or rel.is_absolute() or rel.parts[:2] not in [('experiments','2b-short-event-v1'),('runs','2b-short-event-inputs-20261020-01')]:raise ValueError('member allowlist')
  total+=m.size
  if total>5*2**20:raise ValueError('transfer bound')
  dst=p/str(rel)
  if not dst.resolve().is_relative_to(p):raise ValueError('path escape')
  dst.parent.mkdir(parents=True,exist_ok=True)
  data=tar.extractfile(m).read()
  with dst.open('xb') as f:f.write(data)
  items[str(rel)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(dict(files=items,total_bytes=total)))
"""
    with archive.open('rb') as stream:
        r=subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(code)],
                         stdin=stream,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    (LOCAL/'transfer_stdout.json').write_bytes(r.stdout)
    (LOCAL/'transfer_stderr.txt').write_bytes(r.stderr)
    if r.returncode:raise RuntimeError('transport failed: '+r.stderr.decode('utf-8',errors='replace')[-1600:])
    receipt=json.loads(r.stdout)
    if receipt['files']!=expected:raise ValueError('all transferred bytes must match')
    save(LOCAL/'transfer_receipt.json',dict(files=len(expected),bytes=receipt['total_bytes'],SHA256_equal=True))
    print(json.dumps(dict(files=len(expected),bytes=receipt['total_bytes'])))

def cpu():
    code="""import pathlib,subprocess,os,json,time
p=pathlib.Path.cwd();run=p/'runs/2b-short-event-inputs-20261020-01'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
start=time.monotonic()
with (run/'cpu.log').open('x') as log:
 result=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(p/'experiments/2b-short-event-v1/evaluate.py'),'cpu','--run',str(run)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
proof=run/'cpu_preflight.json'
print(json.dumps(dict(exit_code=result.returncode,seconds=time.monotonic()-start,proof=json.loads(proof.read_text()) if proof.exists() else None,log_tail=(run/'cpu.log').read_text()[-3500:])))
"""
    s=command(code,timeout=330)
    (LOCAL/'remote_cpu_execution.json').write_text(s,encoding='utf-8')
    result=json.loads(s)
    print(json.dumps({k:v for k,v in result.items() if k!='proof'}))
    if result['exit_code']!=0:raise RuntimeError('CPU preflight failed; preserve receipt')
    save(LOCAL/'cpu_preflight.json',result['proof'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['deploy','cpu']);a=p.parse_args()
    globals()[a.mode]()
