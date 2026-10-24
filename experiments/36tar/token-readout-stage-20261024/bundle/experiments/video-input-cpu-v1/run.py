"""Sequential CPU-only bounded run; never submits jobs or retries cases."""
import os,subprocess,sys,time
from common import ROOT,OUT,HERE,read,save

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert not (OUT/'run_intent.json').exists()
save(OUT/'run_intent.json',dict(cases=32,max_wall_seconds=1800,GPU_jobs=0,no_retry=True))
started=time.monotonic();cases=read(OUT/'inputs/cases.json');done=[]
for case in cases:
    remaining=1800-(time.monotonic()-started);assert remaining>1
    with (OUT/('case-'+str(case['index'])+'.log')).open('x') as log:
        result=subprocess.run([sys.executable,str(HERE/'worker.py'),'--case',str(case['index'])],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,timeout=min(240,remaining))
    assert result.returncode==0,dict(case=case,exit_code=result.returncode)
    receipt=read(OUT/('case-'+str(case['index'])+'.json'));assert receipt['status']=='passed'
    done.append(receipt)
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<8*2**20
    print(dict(completed=len(done),total=32,config=case['config'],seconds=round(receipt['input_wall_seconds'],3)),flush=True)
save(OUT/'completed.json',dict(status='complete',cases=32,wall_seconds=time.monotonic()-started,
                             pretrained_loads=0,GPU_jobs=0,local_CPU_only=True))
