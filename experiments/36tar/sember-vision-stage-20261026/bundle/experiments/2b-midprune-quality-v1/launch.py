"""One separately authorized allocation; sequential workers, no resume or retry."""
import os
from pathlib import Path
import subprocess
import sys
import time
from quality_common import ROOT,OUT,read,save,sha,verify


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit new GPU approval and Slurm allocation required')
    protocol=verify();auth=read(OUT/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(OUT/'inputs/execution_lock.json')
    assert auth['max_minutes']==protocol['resource_budget']['minutes']==120
    assert read(OUT/'CPU_cluster_comparison.json')['status']=='passed'
    assert read(OUT/'remote_media_preflight.json')['status']=='passed'
    out=ROOT/'runs'/('2b-midprune-quality-'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    deadline=time.time()+protocol['resource_budget']['minutes']*60-180;completed=[]
    for i,task in enumerate(read(OUT/'inputs/tasks.json')):
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'--task',str(i),
            '--out',str(out/task['name']),'--deadline',str(deadline),'--run-root',str(out)],check=True,cwd=ROOT)
        completed.append(read(out/task['name']/'completed.json'))
    for field,expected in [('pretrained_loads',17),('answers',2140),('warmup_answers',20)]:
        assert sum(c[field] for c in completed)==expected
    assert sum(c['vision_encodings']+c['warmup_encodings'] for c in completed)==37014
    generated=sum(c['generated_tokens']+c['warmup_generated_tokens'] for c in completed)
    assert generated<=protocol['max_generated_tokens']
    save(out/'completed.json',dict(status='complete',processes=len(completed),answers=2140,warmup_answers=20,
        generated_tokens=generated,vision_encodings=37014,pretrained_loads=17,
        execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':main()
