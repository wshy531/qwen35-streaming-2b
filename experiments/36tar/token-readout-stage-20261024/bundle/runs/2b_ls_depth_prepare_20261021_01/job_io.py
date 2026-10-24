"""Scoped read-only progress and verified result download for approved217894."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
JOB='217894'
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import command,ssh,PROJECT,PY


def status():
    code="""import collections,datetime,json,pathlib,subprocess
p=pathlib.Path.cwd().resolve();assert p.is_relative_to(pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
run=p/'runs/2b-ls-depth-217894';prep=p/'runs/2b_ls_depth_prepare_20261021_01'
queue=subprocess.run(['squeue','-j','217894','-h','-o','%i %T %M %R'],capture_output=True,text=True)
account=subprocess.run(['sacct','-j','217894','-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,Timelimit,AllocTRES,MaxRSS'],capture_output=True,text=True)
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'217894','queue':queue.stdout.strip(),'queue_stderr':queue.stderr.strip(),'sacct':account.stdout.strip(),'sacct_stderr':account.stderr.strip()}
if run.exists():
 result['files']={f.name:f.stat().st_size for f in run.iterdir() if f.is_file()}
 for name in ['loaded.json','completed.json']:
  f=run/name
  if f.exists():result[name]=json.loads(f.read_text())
 f=run/'answers.jsonl'
 if f.exists():
  counts=collections.Counter();answers=0;last=None
  with f.open() as stream:
   for line in stream:
    if not line.endswith(chr(10)):continue
    r=json.loads(line);counts[r['arm']]+=1;answers+=1;last=r['id']
  result.update(answers=answers,arms=dict(counts),last_question=last)
for suffix in ['log','err']:
 f=prep/('slurm-217894.'+suffix)
 if f.exists():
  with f.open('rb') as stream:stream.seek(max(0,f.stat().st_size-2000));result['slurm_'+suffix+'_tail']=stream.read().decode(errors='replace')
print(json.dumps(result))
"""
    result=command(code,timeout=45)
    with (HERE/'job_status.jsonl').open('a',encoding='utf-8') as stream:stream.write(result.strip()+'\n')
    print(result)


def collect():
    target=HERE/'results-217894';assert not target.exists()
    archive=HERE/'result_transfer-217894.tar'
    remote="""import hashlib,io,json,pathlib,subprocess,sys,tarfile
p=pathlib.Path.cwd().resolve();assert p.is_relative_to(pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
run=p/'runs/2b-ls-depth-217894';prep=p/'runs/2b_ls_depth_prepare_20261021_01'
assert json.loads((run/'completed.json').read_text())['status']=='complete'
account=subprocess.run(['sacct','-j','217894','-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,Timelimit,AllocTRES,MaxRSS'],capture_output=True,text=True,check=True).stdout
primary=[r.split('|') for r in account.splitlines() if r.split('|')[0]=='217894'];assert len(primary)==1 and primary[0][1:3]==['COMPLETED','0:0'],account
files={name:run/name for name in ['answers.jsonl','events.jsonl','tokens.jsonl','completed.json','loaded.json']}
files.update({'slurm-217894.'+suffix:prep/('slurm-217894.'+suffix) for suffix in ['log','err']})
assert all(f.is_file() and f.resolve().is_relative_to(p) for f in files.values())
assert sum(f.stat().st_size for f in files.values())<96*2**20
manifest={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for name,f in files.items():
  data=f.read_bytes();manifest[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
 data=json.dumps({'files':manifest,'sacct':account,'GPU_submissions':0}).encode();info=tarfile.TarInfo('TRANSFER_MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
"""
    with archive.open('xb') as output:
        result=subprocess.run(ssh()+['cd '+PROJECT+' && PYTHONDONTWRITEBYTECODE=1 '+PY+' -c '+shlex.quote(remote)],
            stdout=output,stderr=subprocess.PIPE,timeout=180)
    (HERE/'collect_stderr.txt').write_bytes(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    allowed={'answers.jsonl','events.jsonl','tokens.jsonl','completed.json','loaded.json','slurm-217894.log','slurm-217894.err','TRANSFER_MANIFEST.json'}
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len(members)==len(allowed) and {m.name for m in members}==allowed
        manifest=json.loads(tar.extractfile('TRANSFER_MANIFEST.json').read())
        for member in members:
            assert member.isfile() and member.size<96*2**20 and '/' not in member.name and '\\' not in member.name
            data=tar.extractfile(member).read()
            if member.name!='TRANSFER_MANIFEST.json':assert manifest['files'][member.name]==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        target.mkdir()
        for member in members:
            with (target/member.name).open('xb') as output:output.write(tar.extractfile(member).read())
    print(dict(files=len(members),bytes=sum(r['bytes'] for r in manifest['files'].values()),all_SHA256_equal=True,sacct=manifest['sacct']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['status','collect']);args=parser.parse_args();globals()[args.mode]()
