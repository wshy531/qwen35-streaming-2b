"""Only bounded project-owned source/metadata transfer and CPU checks."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import ssh, command, PROJECT, PY


def deploy():
    files = [p for p in (ROOT/'experiments/2b-ls-depth-v1').iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md')]
    files += [p for p in (OUT/'inputs').iterdir() if p.is_file()]
    files += [OUT/'CPU_preflight.json']
    assert all('authorization' not in p.name.lower() for p in files)
    assert sum(p.stat().st_size for p in files) < 4*2**20
    archive = OUT/'source_transfer.tar'
    with tarfile.open(archive,'x') as tar:
        for path in files: tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    remote = """import hashlib,json,pathlib,sys,tarfile
p=pathlib.Path.cwd().resolve();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert p.is_relative_to(allowed) and p.name=='qwen35-streaming-2b-36tar'
expected_prefixes={('experiments','2b-ls-depth-v1'),('runs','2b_ls_depth_prepare_20261021_01')}
files={};total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for item in tar:
  rel=pathlib.PurePosixPath(item.name)
  assert not rel.is_absolute() and '..' not in rel.parts and item.isfile() and item.size<3*2**20
  assert rel.parts[:2] in expected_prefixes and 'authorization' not in item.name.lower()
  dst=p/str(rel);assert dst.resolve().is_relative_to(p)
  data=tar.extractfile(item).read();assert len(data)==item.size
  total+=len(data);assert total<4*2**20
  digest=hashlib.sha256(data).hexdigest()
  if dst.exists(): assert dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest()==digest,'refuse overwrite: '+str(rel)
  else:
   dst.parent.mkdir(parents=True,exist_ok=True)
   with dst.open('xb') as output: output.write(data)
  files[str(rel)]={'bytes':len(data),'sha256':digest}
print(json.dumps({'files':files,'bytes':total,'GPU_submissions':0}))
"""
    with archive.open('rb') as stream:
        result = subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(remote)],
            stdin=stream,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    (OUT/'deploy_stderr.txt').write_bytes(result.stderr)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf-8')[-2500:])
    receipt = json.loads(result.stdout)
    expected = {p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    assert receipt['files'] == expected
    with (OUT/'deploy_receipt.json').open('x',encoding='utf-8') as output: json.dump(receipt,output,indent=2)
    print(dict(files=len(files),bytes=receipt['bytes'],all_remote_SHA_equal=True,GPU_submissions=0))


def cpu():
    remote = """import json,os,pathlib,subprocess,time
p=pathlib.Path.cwd().resolve();assert p.is_relative_to(pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
run=p/'runs/2b_ls_depth_prepare_20261021_01'
import hashlib
inp=run/'inputs';lock=json.loads((inp/'execution_lock.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for rel,h in lock['runtime_sha256'].items(): assert sha(p/rel)==h,rel
for rel,h in lock['input_sha256'].items(): assert sha(inp/rel)==h,rel
assert sha(inp/'protocol.json')==lock['protocol_sha256']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
started=time.monotonic()
with (run/'cpu_cluster.log').open('x') as log:
 result=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(p/'experiments/2b-ls-depth-v1/preflight.py')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
print(json.dumps({'exit_code':result.returncode,'seconds':time.monotonic()-started,'log_tail':(run/'cpu_cluster.log').read_text()[-5000:],'GPU_submissions':0}))
"""
    # The deployed local receipt must be kept; preflight writes its own new receipt.
    stage = """import pathlib
p=pathlib.Path.cwd();run=p/'runs/2b_ls_depth_prepare_20261021_01'
src=run/'CPU_preflight.json';dst=run/'CPU_local_receipt.json'
assert src.is_file() and not dst.exists();src.rename(dst)
print('local receipt preserved; no model/GPU started')
"""
    print(command(stage,timeout=30))
    result = command(remote,timeout=270)
    with (OUT/'cluster_CPU_execution.json').open('x',encoding='utf-8') as stream: stream.write(result)
    print(result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['deploy','cpu']);args=parser.parse_args()
    globals()[args.mode]()
