"""Sequential independently imported official versions. Requires an exact future approval receipt."""
import hashlib
import os
import subprocess
import time
from audit_common import ROOT,OUT,read,save
from evaluate import verify


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('refuse model load without an explicitly approved Slurm allocation')
    verify()
    approval=read(OUT/'GPU_authorization.json')
    lock_sha=hashlib.sha256((OUT/'inputs/execution_lock.json').read_bytes()).hexdigest()
    assert approval['approved'] is True and approval['execution_lock_sha256']==lock_sha
    assert approval['max_minutes']==60 and approval['GPUs']==1
    previous=read(ROOT/'runs/2b-resolution-cost-218537/completed.json')
    assert previous['status']=='complete', 'finish the independent resolution/cost run before this profile'
    job=os.environ['SLURM_JOB_ID'];assert job.isdigit()
    base=ROOT/'runs'/('2b-native-audit-'+job);base.mkdir(exist_ok=False)
    deadline=time.time()+60*60-180
    for condition in read(OUT/'inputs/design.json')['conditions']:
        version='qwen35-2b-fp32' if condition.startswith('B0_') else 'native-audit-transformers-5.4.0'
        executable=ROOT/'envs'/version/'bin/python'
        subprocess.run([str(executable),str(ROOT/'experiments/2b-native-audit-v1/evaluate.py'),
            '--condition',condition,'--out',str(base/condition),'--deadline',str(deadline)],check=True,
            timeout=max(1,deadline-time.time()))
    save(base/'completed.json',dict(status='complete',job_id=job,conditions=3,formal_answers=72,
        warmup_answers=3,pretrained_loads=3))


if __name__=='__main__':main()
