"""Scoped transport. No credentials read; SSH uses the user's configured agent."""
import argparse,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LOCAL=ROOT/'runs/2b_natural_eval_20261020_01'
PROJECT='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
PY=PROJECT+'/envs/qwen35-2b-fp32/bin/python'
def ssh():return ['C:/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i',str(Path.home()/'.ssh/qwen35_cluster_temp'),'ChaoSuan']
def command(code,timeout=90):
    p=subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr[:2000]+'\n'+p.stdout[-3000:])
    return p.stdout
def deploy():
    manifest=json.loads((LOCAL/'media_manifest.json').read_text());tarpath=LOCAL/'transfer.tar'
    with tarfile.open(tarpath,'x') as tar:
        for p in HERE.rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh','.json']:tar.add(p,arcname='experiments/2b-natural-eval-v1/'+p.relative_to(HERE).as_posix(),recursive=False)
        for name in ['requests.json','protocol.json','source_lock.json','reuse_provenance.json','media_manifest.json']:
            tar.add(LOCAL/name,arcname='runs/2b-natural-eval-inputs-20261020-01/'+name,recursive=False)
        for item in manifest:tar.add(ROOT/item['local_path'],arcname=item['remote_path'],recursive=False)
    remote="""import hashlib,json,pathlib,sys,tarfile
p=pathlib.Path.cwd();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
if not p.resolve().is_relative_to(allowed):raise ValueError('outside authorized project')
written=[];total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for member in tar:
  rel=pathlib.PurePosixPath(member.name)
  if member.name.startswith('/') or '..' in rel.parts or not member.isfile() or member.size>700*2**20:raise ValueError('unsafe archive member')
  if rel.parts[:2] not in [('experiments','2b-natural-eval-v1'),('runs','2b-natural-eval-inputs-20261020-01'),('data','fixed-natural-eval-v1')]:raise ValueError('unexpected output prefix')
  dst=p/str(rel)
  if not dst.resolve().is_relative_to(p.resolve()):raise ValueError('escape')
  dst.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
  with dst.open('xb') as out:
   source=tar.extractfile(member)
   for block in iter(lambda:source.read(1048576),b''):out.write(block);h.update(block);n+=len(block)
  if n!=member.size:raise ValueError('incomplete file')
  total+=n
  if total>600*2**20:raise ValueError('bounded transfer exceeded')
  written.append(dict(path=str(rel),bytes=n,sha256=h.hexdigest()))
print(json.dumps(dict(status='transferred',files=written,total_bytes=total)))
"""
    # Code is passed as an argument, binary stdin is only our bounded archive.
    import shlex
    with tarpath.open('rb') as f:p=subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(remote)],stdin=f,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
    (LOCAL/'transfer_stderr.txt').write_bytes(p.stderr)
    if p.returncode:raise RuntimeError(p.stderr.decode()[-3000:])
    (LOCAL/'transfer_receipt.json').write_bytes(p.stdout)
    receipt=json.loads(p.stdout);expected={}
    with tarfile.open(tarpath) as tar:
        for m in tar:
            expected[m.name]=(m.size,hashlib.sha256(tar.extractfile(m).read()).hexdigest())
    if any(expected[x['path']]!=(x['bytes'],x['sha256']) for x in receipt['files']) or len(expected)!=len(receipt['files']):raise ValueError('transfer identity mismatch')
    print(json.dumps(dict(files=len(expected),bytes=receipt['total_bytes'],all_remote_SHA256_equal=True)))
def cpu():
    remote=f'''import os,pathlib,subprocess,json,time
p=pathlib.Path.cwd();code=p/'experiments/2b-natural-eval-v1/evaluate.py';run=p/'runs/2b-natural-eval-inputs-20261020-01'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
results=[]
for stage in ['native','memory']:
 start=time.monotonic()
 with (run/('cpu_'+stage+'.log')).open('x') as log:r=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(code),'cpu',stage,'--run',str(run)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=420)
 results.append(dict(stage=stage,exit_code=r.returncode,seconds=time.monotonic()-start))
 if r.returncode:break
print(json.dumps(results))
'''
    s=command(remote,timeout=900);(LOCAL/'cpu_execution.json').write_text(s,encoding='utf-8');print(s)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['deploy','cpu']);a=p.parse_args();globals()[a.mode]()
