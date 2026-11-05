"""One authorized bounded submission, or read-only observation of its own ID."""
import argparse,base64,json
from datetime import datetime
from pathlib import Path
from cluster_io import command,LOCAL,PROJECT
def save(name,s):(LOCAL/name).write_text(s,encoding='utf-8')
def submit():
    with (LOCAL/'submission_attempt.json').open('x',encoding='utf-8') as f:json.dump(dict(authorized_by='user two-hour evaluation window 2026-10-20',maximum_submission_attempts=1),f)
    code=r'''import datetime,json,os,pathlib,subprocess,hashlib,shlex,time
p=pathlib.Path.cwd();expected=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
if p.resolve()!=expected.resolve():raise ValueError('wrong project')
run=p/'runs/2b-natural-eval-inputs-20261020-01';script=p/'experiments/2b-natural-eval-v1/run_eval.sh'
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
lock=sha(run/'source_lock.json');protocol=sha(run/'protocol.json')
proofs=[]
for stage in ['native','memory']:
 proof=json.loads((run/('cpu_'+stage)/'preflight.json').read_text())
 if proof['status']!='passed' or proof['source_lock_sha256']!=lock or proof['protocol_sha256']!=protocol:raise ValueError('CPU preflight missing/drift')
 proofs.append(proof)
if subprocess.check_output(['squeue','-u','u2025010462','--noheader'],text=True).strip():raise ValueError('existing own allocation')
names=['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS']
overrides={k:os.environ[k] for k in names if k in os.environ}
if overrides:raise ValueError('unexpected scheduler override')
subprocess.run(['sh','-n',str(script)],check=True)
deadline=int(datetime.datetime(2026,10,20,6,11,50,tzinfo=datetime.timezone.utc).timestamp())
if deadline-time.time()<3900:raise ValueError('insufficient window for 65 minute job')
wrapper='seconds=$((%s-$(date -u +%%s))); test "$seconds" -gt 120; exec timeout "$seconds" /bin/sh %s' %(deadline,shlex.quote(str(script)))
args=['sbatch','--parsable','--job-name=q35-2b-natural-v1','--partition=g078t','--comment=gx_llm','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=24G','--time=01:05:00','--no-requeue','--chdir='+str(p),'--output='+str(run/'slurm-%j.out'),'--error='+str(run/'slurm-%j.err'),'--wrap',wrapper]
with (run/'submission_attempt.json').open('x') as f:json.dump(dict(arguments=args,command=shlex.join(args),script_sha256=sha(script),proofs=proofs,deadline_UTC='2026-10-20T06:11:50Z'),f,indent=2)
r=subprocess.run(args,capture_output=True,text=True,timeout=30)
result=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,arguments=args,script_sha256=sha(script))
if r.returncode==0:
 job=r.stdout.strip().split(';')[0]
 if not job.isdigit():raise ValueError('unexpected job ID')
 result['job_id']=job;result['allocation']=subprocess.check_output(['scontrol','show','job','-dd',job],text=True)
(run/'submission_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''
    s=command(code);save('submission_result.json',s);print(s)
def observe():
    job=json.loads((LOCAL/'submission_result.json').read_text())['job_id']
    code="job="+repr(job)+r'''
import json,pathlib,subprocess
p=pathlib.Path.cwd();out=p/('runs/2b-natural-eval-'+job);run=p/'runs/2b-natural-eval-inputs-20261020-01'
result=dict(job_id=job,queue=subprocess.check_output(['squeue','-j',job,'--noheader','-o','%i|%T|%M|%l|%R'],text=True),accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS'],text=True))
for stage in ['native','memory']:
 d=out/stage;f=d/'predictions.jsonl';log=out/(stage+'.log')
 result[stage]=dict(predictions=sum(1 for _ in f.open()) if f.exists() else 0,completed=json.loads((d/'completed.json').read_text()) if (d/'completed.json').exists() else None,tail=log.read_text()[-1800:] if log.exists() else '')
result['scheduler_stderr']=(run/('slurm-'+job+'.err')).read_text()[-1500:] if (run/('slurm-'+job+'.err')).exists() else ''
print(json.dumps(result))
'''
    s=command(code);name='observation_'+datetime.now().strftime('%H%M%S')+'.json';save(name,s);print(s)
def fetch():
    job=json.loads((LOCAL/'submission_result.json').read_text())['job_id']
    code="job="+repr(job)+r'''
import base64,hashlib,json,pathlib
p=pathlib.Path.cwd();files=[];total=0
for folder,prefix in [(p/('runs/2b-natural-eval-'+job),'result'),(p/'runs/2b-natural-eval-inputs-20261020-01','input')]:
 for src in sorted(folder.rglob('*')):
  if not src.is_file() or src.is_symlink():continue
  if not src.resolve().is_relative_to(folder.resolve()):raise ValueError('unsafe result path')
  value=src.read_bytes();total+=len(value)
  if len(value)>32*2**20 or total>64*2**20:raise ValueError('bounded evidence download exceeded')
  files.append(dict(path=prefix+'/'+src.relative_to(folder).as_posix(),bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),base64=base64.b64encode(value).decode()))
print(json.dumps(dict(files=files,bytes=total)))
'''
    result=json.loads(command(code,timeout=180));out=LOCAL/('gpu_'+job);out.mkdir(exist_ok=False)
    import hashlib
    for item in result['files']:
        dest=out/item['path'];value=base64.b64decode(item.pop('base64'))
        if not dest.resolve().is_relative_to(out.resolve()) or hashlib.sha256(value).hexdigest()!=item['sha256']:raise ValueError('evidence integrity')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(value)
    save('fetch_receipt.json',json.dumps(result,indent=2));print(json.dumps(dict(files=len(result['files']),bytes=result['bytes'],output=str(out))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['submit','observe','fetch']);a=p.parse_args();globals()[a.mode]()
