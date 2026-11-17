import os,subprocess
from adapt import *

def main():
    assert os.name=='posix';verify_adapt()
    auth=read(HERE/'authorization.json');assert auth['approved'] and auth['lock_sha']==sha(HERE/'lock.json')
    a=read(HERE/'CPU_local.json');b=read(HERE/'CPU_cluster_v2.json')
    for k in ('vision','micro_answers','old_recipe_bitwise_equal','pruned_descriptor_checks','lock_sha'):
        assert a[k]==b[k],k
    assert a['adapt_passed'] and b['adapt_passed'] and read(HERE/'remote_check.json')['passed']
    script=Path(__file__).parent/'run_adapt.sh';assert b'\r' not in script.read_bytes()
    subprocess.run(['bash','-n',str(script)],check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    t=subprocess.run(['sbatch','--test-only','--no-requeue',str(script)],env=env,text=True,capture_output=True)
    save(HERE/'test_only.json',dict(returncode=t.returncode,stdout=t.stdout,stderr=t.stderr));assert t.returncode==0
    cmd=['sbatch','--parsable','--no-requeue','--output=runs/7b-pruned-head-adapt-%j.log','--error=runs/7b-pruned-head-adapt-%j.log',str(script)]
    with (HERE/'submission_claim.json').open('x') as f:json.dump(dict(command=cmd,authorization=auth),f)
    v=subprocess.run(cmd,env=env,cwd=ROOT,text=True,capture_output=True)
    r=dict(returncode=v.returncode,stdout=v.stdout.strip(),stderr=v.stderr.strip())
    save(HERE/'submission_receipt.json',r);print(json.dumps(r));assert v.returncode==0
if __name__=='__main__':main()
