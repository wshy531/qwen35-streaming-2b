"""Only after specific approval and clean completion of job 218526; never resubmit."""
import os
from pathlib import Path
import subprocess
import sys
import time
from resolution_common import ROOT,OUT,read,save,sha,verify


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit Slurm GPU approval required')
    protocol=verify()
    auth=read(OUT/'GPU_authorization.json')
    assert auth['approved'] is True and auth['execution_lock_SHA']==sha(OUT/'inputs/execution_lock.json')
    assert not subprocess.check_output(['squeue','-h','-j','218526'],text=True).strip(),'prior job still active'
    accounting=subprocess.check_output(['sacct','-j','218526','--format=JobIDRaw,State,ExitCode','-nP'],text=True)
    assert '218526|COMPLETED|0:0' in accounting
    for mode in ('bounded','reference'):
        assert read(ROOT/'runs/2b-sember-main-218526'/mode/'completed.json')['status']=='complete'
    assert read(OUT/'remote_CPU_preflight.json')['status']=='passed'
    assert read(OUT/'CPU_comparison.json')['status']=='passed'
    out=ROOT/'runs'/('2b-resolution-cost-'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    deadline=time.time()+protocol['resource_budget']['minutes']*60-180
    completed=[]
    for index,task in enumerate(read(OUT/'inputs/tasks.json')):
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'--task',str(index),
            '--out',str(out/task['name']),'--deadline',str(deadline),'--run-root',str(out)],check=True,cwd=ROOT)
        completed.append(read(out/task['name']/'completed.json'))
    assert sum(c['answers']+c['warmup_answers'] for c in completed)==protocol['max_answers']
    assert sum(c['vision_encodings']+c['warmup_encodings'] for c in completed)==protocol['max_vision_encodings']
    assert sum(c['pretrained_loads'] for c in completed)==11
    assert sum(c['generated_tokens']+c['warmup_generated_tokens'] for c in completed)<=protocol['max_generated_tokens']
    save(out/'completed.json',dict(status='complete',processes=len(completed),
        answers=sum(c['answers'] for c in completed),warmup_answers=sum(c['warmup_answers'] for c in completed),
        vision_encodings=sum(c['vision_encodings']+c['warmup_encodings'] for c in completed),
        execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':main()
