import os,subprocess
from combination import *

def main():
    assert os.name=='posix';verify_all()
    auth=read(HERE/'authorization.json');assert auth['approved'] and auth['lock_sha']==sha(HERE/'lock.json')
    a=read(HERE/'CPU_local.json');b=read(HERE/'CPU_cluster.json')
    for k in ('vision','micro_answers','frozen_head_checks','questions','references','lock_sha'):assert a[k]==b[k],k
    assert a['combination_passed'] and b['combination_passed'] and read(HERE/'remote_check.json')['passed']
    script=Path(__file__).parent/'run_eval.sh';assert b'\r' not in script.read_bytes()
    subprocess.run(['bash','-n',str(script)],check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    t=subprocess.run(['sbatch','--test-only','--no-requeue',str(script)],env=env,text=True,capture_output=True)
    save(HERE/'test_only.json',dict(returncode=t.returncode,stdout=t.stdout,stderr=t.stderr));assert t.returncode==0
    cmd=['sbatch','--parsable','--no-requeue','--output=runs/7b-prune-head-%j.log','--error=runs/7b-prune-head-%j.log',str(script)]
    with (HERE/'submission_claim.json').open('x') as f:json.dump(dict(command=cmd,authorization=auth),f)
    v=subprocess.run(cmd,env=env,cwd=ROOT,text=True,capture_output=True)
    receipt=dict(returncode=v.returncode,stdout=v.stdout.strip(),stderr=v.stderr.strip())
    save(HERE/'submission_receipt.json',receipt);print(json.dumps(receipt));assert v.returncode==0
if __name__=='__main__':main()
