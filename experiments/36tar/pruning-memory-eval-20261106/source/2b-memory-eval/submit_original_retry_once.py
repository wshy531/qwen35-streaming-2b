"""Execute ONLY the explicitly reauthorized original 20-minute acceptance job.

Uses a local exclusive marker before network access; no automatic resubmission.
The user was informed of resources/script/project immediately before this call.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess


REMOTE=r'''
import getpass,hashlib,json,os,re,subprocess
from pathlib import Path
p=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
if Path.cwd().resolve()!=p.resolve() or not p.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve()) or getpass.getuser()!='u2025010462':raise ValueError('remote scope mismatch')
if any(k in os.environ for k in ['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS']) or os.environ.get('SLURM_JOB_ID'):raise ValueError('unexpected Slurm override/allocation')
if subprocess.check_output(['git','-C',str(p/'repo'),'branch','--show-current'],text=True).strip()!='36tar/2b-memory-eval':raise ValueError('wrong branch')
if subprocess.check_output(['squeue','-u','u2025010462','--noheader'],text=True).strip():raise ValueError('own queue no longer empty')
script=p/'repo/experiments/36tar/2b-memory-eval/run_first_2b_baseline.sh'
if hashlib.sha256(script.read_bytes()).hexdigest()!='21731748f40182c80ebba78e4b1ac7ba4942d5590f60c3c65882cd93cd1a91c8':raise ValueError('run script changed')
args=['sbatch','--parsable','--partition=g078t','--comment=gx_llm','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=24G','--time=00:20:00','--job-name=qwen2b-first','--chdir='+str(p),'--output='+str(p/'logs/first-2b-%j.out'),'--error='+str(p/'logs/first-2b-%j.err'),str(script)]
env=dict(os.environ,CUDA_CACHE_PATH=str(p/'cache/cuda'))
r=subprocess.run(args,capture_output=True,text=True,env=env)
job=r.stdout.strip().split(';')[0] if r.returncode==0 else None
if job is not None and not re.fullmatch(r'\d+',job):raise ValueError('unrecognized successful sbatch response; no retry')
print(json.dumps(dict(status='actual_single_submission_attempt',exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,job_id=job,command=args,allocated_resources_verified=False)))
'''


def main():
    root=Path(__file__).resolve().parents[2]
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    out=Path(parser.parse_args().run_dir).resolve()
    if out.parent!=(root/'runs').resolve():raise ValueError('run directory must be directly under project runs')
    pre=json.loads((out/'preflight_connection.json').read_text())
    if not pre['preflight_ok']:raise ValueError('read-only preflight did not pass')
    if (datetime.now(timezone.utc).timestamp()-(out/'preflight_connection.json').stat().st_mtime)>900:
        raise ValueError('preflight stale; do not submit')
    with (out/'submission_attempt_marker.json').open('x',encoding='utf-8') as f:
        json.dump(dict(attempt_utc=datetime.now(timezone.utc).isoformat(),maximum_attempts=1,
            authorization='User requested another GPU submission attempt; same previously specified20-minute interface budget re-disclosed. No expanded benchmark.',
            budget=dict(gpus=1,cpus=4,ram_gib=24,minutes=20,partition='g078t',comment='gx_llm')),f,indent=2)
    (out/'single_submission_remote.py').write_text(REMOTE,encoding='utf-8')
    project='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
    args=['C:/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
        '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i',str(Path.home()/'.ssh/qwen35_cluster_temp'),
        'ChaoSuan','cd '+project+' && PYTHONDONTWRITEBYTECODE=1 '+project+'/envs/qwen35-2b-fp32/bin/python -']
    try:
        proc=subprocess.run(args,input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=60)
    except subprocess.TimeoutExpired:
        result=dict(status='submission_connection_timed_out_unknown_remote_state',automatic_retry=False)
        (out/'submission_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result));return
    (out/'submission_ssh_stderr.txt').write_text(proc.stderr,encoding='utf-8')
    if proc.returncode==0:
        result=json.loads(proc.stdout)
        import shlex
        (out/'actual_sbatch_command.txt').write_text('cd '+project+' && CUDA_CACHE_PATH='+project+'/cache/cuda '+shlex.join(result['command'])+'\n',encoding='utf-8')
    else:
        result=dict(status='SSH_or_remote_guard_failed_no_automatic_retry',ssh_exit=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,job_id=None)
    result['automatic_retry']=False
    (out/'submission_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
