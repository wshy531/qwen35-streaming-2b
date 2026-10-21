"""Bounded project-only deployment/CPU check; never submits GPU jobs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from prepare import ROOT, OUT, read, save, sha

sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import PROJECT, PY, ssh, command

CODE = 'experiments/2b-validation-T-v1'
INPUT = 'runs/2b-validation-T-inputs-20261020-01'
DATA = 'data/T-new40-20261020'


def deploy():
    expected,files = {},[]
    for name in ['evaluate.py','run_eval.sh']:
        files.append((Path(__file__).parent/name,CODE+'/'+name))
    for name in ['requests.json','runtime_protocol.json','runtime_lock.json']:
        files.append((OUT/'runtime_inputs'/name,INPUT+'/'+name))
    for row in read(OUT/'download_manifest.json')['files']:
        files.append((ROOT/row['media_path'],DATA+'/'+row['source_path']+'.mp4'))
    for path,destination in files:
        expected[destination] = dict(bytes=path.stat().st_size,sha256=sha(path))
    assert sum(x['bytes'] for x in expected.values()) < 300*2**20
    save(OUT/'deployment_started.json',dict(expected=expected,project=PROJECT,
        byte_cap=300*2**20,wall_seconds=240,automatic_retries=0,GPU_jobs=0))
    remote = """import hashlib,json,pathlib,sys,tarfile
p=pathlib.Path.cwd().resolve();root=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
if not p.is_relative_to(root):raise ValueError('unauthorized project')
result={};total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for m in archive:
  rel=pathlib.PurePosixPath(m.name)
  if not m.isfile() or rel.is_absolute() or '..' in rel.parts or m.size>24*2**20:raise ValueError('unsafe member')
  if rel.parts[:2] not in [('experiments','2b-validation-T-v1'),('runs','2b-validation-T-inputs-20261020-01'),('data','T-new40-20261020')]:raise ValueError('unexpected destination')
  total+=m.size
  if total>300*2**20:raise ValueError('transfer budget')
  dst=p/str(rel)
  if not dst.resolve().is_relative_to(p):raise ValueError('path escape')
  dst.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
  with dst.open('xb') as out:
   src=archive.extractfile(m)
   for chunk in iter(lambda:src.read(1048576),b''):out.write(chunk);h.update(chunk);n+=len(chunk)
  if n!=m.size:raise ValueError('partial member')
  result[str(rel)]=dict(bytes=n,sha256=h.hexdigest())
print(json.dumps(dict(files=result,total_bytes=total)))
"""
    proc = subprocess.Popen(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(remote)],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin,mode='w|') as archive:
            for path,destination in files:
                archive.add(path,arcname=destination,recursive=False)
        proc.stdin.close()
        proc.stdin = None
        stdout,stderr = proc.communicate(timeout=240)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    (OUT/'deployment_stdout.json').write_bytes(stdout)
    (OUT/'deployment_stderr.txt').write_bytes(stderr)
    if proc.returncode:
        raise RuntimeError('project deployment failed; captured output preserved')
    receipt = json.loads(stdout)
    assert receipt['files']==expected
    save(OUT/'deployment_receipt.json',dict(status='all_bytes_SHA_verified',files=len(expected),
        media=40,total_bytes=receipt['total_bytes'],project=PROJECT,temporary_media_archive_created=False,GPU_jobs=0))
    print(read(OUT/'deployment_receipt.json'))


def cpu():
    remote = """import json,os,pathlib,subprocess,time
p=pathlib.Path.cwd();r=p/'runs/2b-validation-T-inputs-20261020-01'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
results=[]
for stage in ['native','memory']:
 start=time.monotonic()
 with (r/('cpu_'+stage+'.log')).open('x') as log:
  code=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(p/'experiments/2b-validation-T-v1/evaluate.py'),'cpu','--run',str(r),'--stage',stage],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=210)
 proof=r/('cpu_'+stage+'_preflight.json')
 results.append(dict(stage=stage,exit_code=code.returncode,seconds=time.monotonic()-start,proof=json.loads(proof.read_text()) if proof.exists() else None,log_tail=(r/('cpu_'+stage+'.log')).read_text()[-2000:]))
 if code.returncode:break
print(json.dumps(results))
"""
    text = command(remote,timeout=450)
    result = json.loads(text)
    save(OUT/'remote_CPU_execution.json',result)
    print(json.dumps(result,ensure_ascii=False))
    assert len(result)==2 and all(x['exit_code']==0 for x in result), 'CPU preflight failed; no GPU submission'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['deploy','cpu'])
    args=parser.parse_args()
    {'deploy':deploy,'cpu':cpu}[args.mode]()
