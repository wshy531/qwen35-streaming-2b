"""Scoped small-source transfer and CPU precheck only; cannot submit GPU jobs."""
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys

from review_media import ROOT,OUT
import hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import ssh,PROJECT,PY,command

REMOTE_INPUT='runs/2b-A04-inputs-20261021-02'


def inspect():
    code="""import getpass,json,pathlib,socket,subprocess
p=pathlib.Path.cwd();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert p.resolve().is_relative_to(allowed) and getpass.getuser()=='u2025010462'
q=subprocess.run(['squeue','-u',getpass.getuser(),'-h','-o','%i %T %j'],capture_output=True,text=True,check=True)
d=subprocess.run(['df','-h',str(p)],capture_output=True,text=True,check=True)
print(json.dumps(dict(project=str(p),resolved=str(p.resolve()),host=socket.gethostname(),user=getpass.getuser(),queue=q.stdout,disk=d.stdout,GPU_submissions=0)))
"""
    result=command(code);(OUT/'remote_inspection.json').write_text(result,encoding='utf-8');print(result)


def deploy():
    import tarfile
    files=[]
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.sh'):files.append((p,p.relative_to(ROOT).as_posix()))
    for p in (OUT/'runtime_inputs_v2').iterdir():files.append((p,REMOTE_INPUT+'/'+p.name))
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as tar:
        for p,rel in files:tar.add(p,arcname=rel,recursive=False)
    payload=buffer.getvalue();assert len(payload)<4*2**20
    code="""import hashlib,json,pathlib,sys,tarfile
p=pathlib.Path.cwd();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert p.resolve().is_relative_to(allowed)
written=[];total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  rel=pathlib.PurePosixPath(m.name)
  assert not rel.is_absolute() and '..' not in rel.parts and m.isfile() and m.size<2*2**20
  assert rel.parts[:2] in [('experiments','2b-spatial-time-v1'),('runs','2b-A04-inputs-20261021-02')]
  dst=p/str(rel);assert dst.resolve().is_relative_to(p.resolve())
  dst.parent.mkdir(parents=True,exist_ok=True)
  data=tar.extractfile(m).read();h=hashlib.sha256(data).hexdigest()
  assert len(data)==m.size;total+=len(data);assert total<4*2**20
  if dst.exists():assert hashlib.sha256(dst.read_bytes()).hexdigest()==h,'different existing file: no overwrite'
  else:
   with dst.open('xb') as f:f.write(data)
  written.append(dict(path=str(rel),bytes=len(data),sha256=h))
print(json.dumps(dict(status='deployed',files=written,total_bytes=total,GPU_submissions=0)))
"""
    r=subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(code)],
        input=payload,capture_output=True,timeout=90)
    (OUT/'deploy_stderr.txt').write_bytes(r.stderr)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-3000:])
    receipt=json.loads(r.stdout);expected={rel:(p.stat().st_size,sha(p)) for p,rel in files}
    assert len(receipt['files'])==len(expected)
    assert all(expected[x['path']]==(x['bytes'],x['sha256']) for x in receipt['files'])
    (OUT/'deploy_receipt.json').write_bytes(r.stdout)
    print(dict(files=len(files),bytes=receipt['total_bytes'],all_remote_SHA_equal=True,GPU_submissions=0))


def cpu():
    code="""import json,os,pathlib,subprocess,time
p=pathlib.Path.cwd();run=p/'runs/2b-A04-inputs-20261021-02'
assert p.resolve().is_relative_to(pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
start=time.monotonic()
with (run/'cpu_cluster.log').open('x') as log:
 r=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(p/'experiments/2b-spatial-time-v1/check_compare.py'),'--run',str(run)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
result=dict(exit_code=r.returncode,seconds=time.monotonic()-start)
if r.returncode:result['log_tail']=(run/'cpu_cluster.log').read_text()[-5000:]
else:result['proof']=json.loads((run/'cpu_cluster.json').read_text())
print(json.dumps(result))
"""
    result=command(code,timeout=110)
    with (OUT/'cluster_CPU_execution.json').open('x',encoding='utf-8') as f:f.write(result)
    receipt=json.loads(result)
    if receipt['exit_code']:raise RuntimeError(receipt['log_tail'])
    assert receipt['proof']['status']=='passed';print(result)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('inspect','deploy','cpu')
    globals()[sys.argv[1]]()
