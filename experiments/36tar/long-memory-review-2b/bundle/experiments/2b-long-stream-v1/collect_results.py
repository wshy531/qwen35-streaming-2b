"""Retrieve only this finished job's small evidence; no media, weights or secrets."""
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from inventory_local import ROOT,OUT
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import PROJECT,PY,ssh
from evaluate import sha


def main():
    remote="""import hashlib,io,json,pathlib,subprocess,sys,tarfile
p=pathlib.Path.cwd();assert p.resolve().is_relative_to(pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
job='217776';queue=subprocess.run(['squeue','-j',job,'-h','-o','%i %T'],capture_output=True,text=True,timeout=15)
assert not queue.stdout.strip(),'job still queued/running; no collection yet'
account=subprocess.run(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList'],capture_output=True,text=True,timeout=15)
base=p/('runs/2b-long-stream-'+job);files=[]
for stage in ['native','memory']:
 folder=base/stage
 if folder.exists():
  for f in folder.iterdir():
   assert f.is_file() and f.suffix in ['.json','.jsonl']
   files.append((f,stage+'/'+f.name))
run=p/'runs/2b-long-stream-inputs-20261020-02'
for name in ['protocol.json','runtime_lock.json','requests.json','sources.json','snapshots.json','expected_layout34.json','local_cpu_proofs.json','cpu_native.json','cpu_memory.json','GPU_authorization.json','submission_attempt.json','submission_receipt.json']:
 files.append((run/name,'input/'+name))
for suffix in ['log','err']:files.append((p/('runs/2b-long-stream-'+job+'.'+suffix),'job.'+suffix))
assert all(not f.is_symlink() and f.resolve().is_relative_to(p.resolve()) for f,_ in files)
assert sum(f.stat().st_size for f,_ in files)<128*2**20
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for f,name in files:
  h=hashlib.sha256()
  with f.open('rb') as source:
   for block in iter(lambda:source.read(1048576),b''):h.update(block)
  manifest.append(dict(path=name,bytes=f.stat().st_size,sha256=h.hexdigest()))
  archive.add(f,arcname=name,recursive=False)
 evidence=json.dumps(dict(files=manifest,queue=queue.stdout,sacct=account.stdout,sacct_stderr=account.stderr)).encode()
 item=tarfile.TarInfo('collection_manifest.json');item.size=len(evidence);archive.addfile(item,io.BytesIO(evidence))
"""
    result=subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(remote)],
        capture_output=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-3000:])
    assert len(result.stdout)<20*2**20
    archive_path=OUT/'gpu_217776_evidence.tar.gz'
    with archive_path.open('xb') as f:f.write(result.stdout)
    dest=OUT/'gpu_217776';dest.mkdir(exist_ok=False)
    total=0
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
        for item in archive:
            rel=Path(item.name)
            assert not rel.is_absolute() and '..' not in rel.parts and item.isfile()
            assert rel.parts[0] in ('native','memory','input','job.log','job.err','collection_manifest.json')
            target=dest/rel;assert target.resolve().is_relative_to(dest.resolve())
            total+=item.size;assert total<128*2**20
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:
                source=archive.extractfile(item)
                for block in iter(lambda:source.read(1048576),b''):f.write(block)
    manifest=json.loads((dest/'collection_manifest.json').read_text(encoding='utf-8'))
    for f in manifest['files']:
        assert (dest/f['path']).stat().st_size==f['bytes'] and sha(dest/f['path'])==f['sha256']
    receipt=dict(files=len(manifest['files']),uncompressed_bytes=total,archive_bytes=archive_path.stat().st_size,
        archive_sha256=sha(archive_path),all_remote_local_sha256_equal=True,
        queue_empty=True,sacct=manifest['sacct'])
    with (OUT/'collection_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(receipt)


if __name__=='__main__':main()
