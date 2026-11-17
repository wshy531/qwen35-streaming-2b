"""Two separately capped jobs covered by the user's explicit GPU work window."""
import argparse,os,time,subprocess
from common import *

def main(mode):
    assert os.name=='posix' and mode in ('prune','selector');verify()
    a=read(PREP/'authorization.json');assert a['approved'] and time.time()<a['submit_before_unix']
    assert read(PREP/'remote_verified.json')['lock_sha256']==sha(PREP/'lock.json')
    local=read(PREP/'CPU_local.json');remote=read(PREP/'CPU_cluster.json')
    assert local['passed'] and remote['passed'] and local['vision']==remote['vision']
    for x,y in zip(local['micro_answers'],remote['micro_answers']):
        for k in ('mode','ids','token_sha','pos_sha'):assert x[k]==y[k],k
    l=read(PREP/'layouts_local.json');r=read(PREP/'layouts_cluster.json')
    assert l==r and l['passed'] and l['layouts']==14848
    script=Path(__file__).parent/f'run_{mode}.sh'
    assert script.read_bytes().startswith(b'#!/bin/bash\n') and b'\r' not in script.read_bytes()
    cmd=['sbatch','--parsable','--no-requeue',f'--output=runs/7b-{mode}-transfer-%j.log',f'--error=runs/7b-{mode}-transfer-%j.log',str(script)]
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    with (PREP/f'{mode}_submission_claim.json').open('x') as f:json.dump(dict(time=time.time(),command=cmd,lock_sha256=sha(PREP/'lock.json')),f)
    r=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
    result=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip(),automatic_retry=False)
    save(PREP/f'{mode}_submission_receipt.json',result);print(json.dumps(result),flush=True)
    assert r.returncode==0 and r.stdout.strip().split(';')[0].isdigit()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prune','selector'],required=True);main(ap.parse_args().mode)
