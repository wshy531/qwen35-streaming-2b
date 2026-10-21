"""One human-authorized job, owned-job observation and bounded evidence fetch."""
import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import sys

from prepare import ROOT,OUT,read,save,sha

sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import command


def submit():
    auth=read(OUT/'GPU_authorization.json')
    protocol=OUT/'runtime_inputs/runtime_protocol.json'
    assert auth['approved'] is True and auth['protocol_sha256']==sha(protocol)
    checks=read(OUT/'remote_CPU_execution.json')
    assert len(checks)==2 and all(c['exit_code']==0 and c['proof']['status']=='passed' and
        c['proof']['protocol_sha256']==sha(protocol) for c in checks)
    save(OUT/'submission_attempt_LOCAL.json',dict(maximum_attempts=1,controller_sha256=sha(__file__),
        authorization_sha256=sha(OUT/'GPU_authorization.json'),protocol_sha256=sha(protocol)))
    code='auth='+repr(auth)+r'''
import pathlib,subprocess,os,json,hashlib,shlex
p=pathlib.Path.cwd();root=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape')
if not p.resolve().is_relative_to(root.resolve()):raise ValueError('outside authorized root')
r=p/'runs/2b-validation-T-inputs-20261020-01';script=p/'experiments/2b-validation-T-v1/run_eval.sh'
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
protocol=json.loads((r/'runtime_protocol.json').read_text())
if auth['approved'] is not True or auth['protocol_sha256']!=sha(r/'runtime_protocol.json'):raise ValueError('specific approval required')
for stage in ['native','memory']:
 proof=json.loads((r/('cpu_'+stage+'_preflight.json')).read_text())
 if proof['status']!='passed' or proof['protocol_sha256']!=auth['protocol_sha256']:raise ValueError('CPU preflight identity')
for name,h in json.loads((r/'runtime_lock.json').read_text()).items():
 if sha(p/name)!=h:raise ValueError('source drift')
overrides={k:os.environ[k] for k in ['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS'] if k in os.environ}
if overrides:raise ValueError('scheduler environment overrides present')
own=subprocess.check_output(['squeue','-u','u2025010462','--noheader'],text=True).strip()
if own:raise ValueError('existing own allocation; no overlapping task')
subprocess.run(['bash','-n',str(script)],check=True)
with (r/'GPU_authorization.json').open('x') as f:json.dump(auth,f,indent=2)
args=['sbatch','--parsable','--chdir='+str(p),'--output='+str(r/'slurm-%j.log'),'--error='+str(r/'slurm-%j.err'),str(script)]
with (r/'submission_attempt.json').open('x') as f:json.dump(dict(arguments=args,command=shlex.join(args),script_sha256=sha(script),protocol_sha256=auth['protocol_sha256'],maximum_attempts=1),f,indent=2)
v=subprocess.run(args,capture_output=True,text=True,timeout=30)
result=dict(exit_code=v.returncode,stdout=v.stdout,stderr=v.stderr,arguments=args,scheduler_environment_overrides=overrides)
if v.returncode==0:
 job=v.stdout.strip().split(';')[0]
 if not job.isdigit():raise ValueError('unexpected job ID; do not resubmit')
 result['job_id']=job;result['allocation']=subprocess.check_output(['scontrol','show','job','-dd',job],text=True)
with (r/'submission_result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
    result=json.loads(command(code))
    save(OUT/'submission_result.json',result)
    print(json.dumps(result))


def observe():
    job=read(OUT/'submission_result.json')['job_id']
    code='job='+repr(job)+r'''
import pathlib,json,subprocess
p=pathlib.Path.cwd();out=p/('runs/2b-validation-T-'+job);r=p/'runs/2b-validation-T-inputs-20261020-01'
q=subprocess.run(['squeue','-j',job,'--noheader','-o','%i|%T|%M|%l|%R'],text=True,capture_output=True)
result=dict(job_id=job,queue=q.stdout,queue_exit=q.returncode,accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS'],text=True))
for stage in ['native','memory']:
 d=out/stage;pred=d/'predictions.jsonl'
 result[stage]=dict(predictions=sum(1 for _ in pred.open()) if pred.exists() else 0,completed=json.loads((d/'completed.json').read_text()) if (d/'completed.json').exists() else None)
result['log_tail']=(r/('slurm-'+job+'.log')).read_text()[-1200:] if (r/('slurm-'+job+'.log')).exists() else ''
result['error_tail']=(r/('slurm-'+job+'.err')).read_text()[-1600:] if (r/('slurm-'+job+'.err')).exists() else ''
print(json.dumps(result))
'''
    result=json.loads(command(code))
    save(OUT/('observation_'+datetime.now().strftime('%H%M%S%f')+'.json'),result)
    print(json.dumps(result))


def fetch():
    job=read(OUT/'submission_result.json')['job_id']
    code='job='+repr(job)+r'''
import pathlib,json,base64,hashlib
p=pathlib.Path.cwd();items=[];total=0
for folder,prefix in [(p/('runs/2b-validation-T-'+job),'result'),(p/'runs/2b-validation-T-inputs-20261020-01','input')]:
 for src in sorted(folder.rglob('*')):
  if not src.is_file() or src.is_symlink() or src.suffix not in ['.json','.jsonl','.log','.err']:continue
  if not src.resolve().is_relative_to(folder.resolve()):raise ValueError('fetch path escape')
  data=src.read_bytes();total+=len(data)
  if len(data)>20*2**20 or total>48*2**20:raise ValueError('small evidence fetch budget')
  items.append(dict(path=prefix+'/'+src.relative_to(folder).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode()))
print(json.dumps(dict(files=items,total_bytes=total)))
'''
    result=json.loads(command(code,timeout=180))
    destination=OUT/('gpu_'+job)
    destination.mkdir(exist_ok=False)
    for item in result['files']:
        data=base64.b64decode(item.pop('base64'))
        path=destination/item['path']
        assert path.resolve().is_relative_to(destination.resolve()) and sha_bytes(data)==item['sha256'] and len(data)==item['bytes']
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    save(OUT/'fetch_receipt.json',result)
    print(dict(files=len(result['files']),bytes=result['total_bytes'],destination=str(destination)))


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['submit','observe','fetch'])
    args=parser.parse_args();globals()[args.mode]()
