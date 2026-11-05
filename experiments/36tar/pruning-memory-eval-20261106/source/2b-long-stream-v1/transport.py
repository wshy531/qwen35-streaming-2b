"""Small, exclusive, project-scoped deployment. No GPU submission capability."""
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from inventory_local import ROOT,OUT
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import ssh,PROJECT,PY,command
from evaluate import sha


def deploy(version=1):
    here=Path(__file__).parent;files=[]
    for p in here.iterdir():
        if p.suffix in ('.py','.sh') and p.name!='transport.py':
            files.append((p,p.relative_to(ROOT).as_posix()))
    input_dir='runtime_inputs' if version==1 else 'runtime_inputs_v2'
    remote_dir='runs/2b-long-stream-inputs-20261020-'+f'{version:02d}'
    for p in (OUT/input_dir).iterdir():files.append((p,remote_dir+'/'+p.name))
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for p,rel in files:archive.add(p,arcname=rel,recursive=False)
    payload=buffer.getvalue();assert len(payload)<4*2**20
    code="""import hashlib,json,pathlib,sys,tarfile
p=pathlib.Path.cwd();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert p.resolve().is_relative_to(allowed)
out=[];total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for member in archive:
  rel=pathlib.PurePosixPath(member.name)
  assert not rel.is_absolute() and '..' not in rel.parts and member.isfile() and member.size<2*2**20
  assert rel.parts[:2] in [('experiments','2b-long-stream-v1'),('runs','2b-long-stream-inputs-20261020-01'),('runs','2b-long-stream-inputs-20261020-02')]
  target=p/str(rel);assert target.resolve().is_relative_to(p.resolve())
  target.parent.mkdir(parents=True,exist_ok=True)
  payload=archive.extractfile(member).read();h=hashlib.sha256(payload);n=len(payload)
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==h.hexdigest(),'existing file differs; no overwrite'
  else:
   with target.open('xb') as f:f.write(payload)
  assert n==member.size;total+=n;assert total<4*2**20
  out.append(dict(path=str(rel),bytes=n,sha256=h.hexdigest()))
print(json.dumps(dict(status='deployed',files=out,total_bytes=total)))
"""
    result=subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(code)],
        input=payload,capture_output=True,timeout=90)
    suffix='' if version==1 else '_v2'
    (OUT/('remote_deploy_stderr'+suffix+'.txt')).write_bytes(result.stderr)
    (OUT/('remote_deploy_stdout'+suffix+'.json')).write_bytes(result.stdout)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-3000:])
    receipt=json.loads(result.stdout);expected={rel:(p.stat().st_size,sha(p)) for p,rel in files}
    assert len(receipt['files'])==len(expected)
    assert all(expected[f['path']]==(f['bytes'],f['sha256']) for f in receipt['files'])
    print(dict(files=len(files),bytes=receipt['total_bytes'],remote_SHA_equal=True,GPU_jobs=0))


def cpu(version=1):
    code="""import json,os,pathlib,subprocess,time
p=pathlib.Path.cwd();run=p/'runs/2b-long-stream-inputs-20261020-01'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
results=[]
for stage in ['native','memory']:
 start=time.monotonic()
 with (run/('cpu_'+stage+'.log')).open('x') as log:
  r=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(p/'experiments/2b-long-stream-v1/cluster_preflight.py'),'--run',str(run),'--stage',stage],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
 result=dict(stage=stage,exit_code=r.returncode,seconds=time.monotonic()-start)
 if r.returncode:
  result['log_tail']=(run/('cpu_'+stage+'.log')).read_text()[-5000:]
 else:result['proof']=json.loads((run/('cpu_'+stage+'.json')).read_text())
 results.append(result)
 if r.returncode:break
print(json.dumps(results))
"""
    if version==2:code=code.replace('inputs-20261020-01','inputs-20261020-02')
    result=command(code,timeout=200)
    with (OUT/('remote_CPU_execution'+('' if version==1 else '_v2')+'.json')).open('x',encoding='utf-8') as f:f.write(result)
    print(result)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('deploy','cpu','deploy2','cpu2')
    mode=sys.argv[1];globals()[mode.rstrip('2')](2 if mode.endswith('2') else 1)
