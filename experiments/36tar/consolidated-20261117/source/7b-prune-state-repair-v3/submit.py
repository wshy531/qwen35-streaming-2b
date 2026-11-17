import os,sys,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-prune-selector-transfer-v1'))
from common import *
from recovery_common import REC,verify_recovery

def main(mode):
    assert os.name=='posix';verify();verify_recovery()
    a=read(REC/'CPU_local.json');b=read(REC/'CPU_cluster.json')
    assert a['passed'] and b['passed'] and a['recovery_contracts'] and b['recovery_contracts']
    assert a['vision']==b['vision'] and a['lock_sha']==b['lock_sha']==sha(REC/'lock.json')
    assert a['micro_answers']==b['micro_answers']
    assert read(REC/'remote_check.json')['passed']
    auth=read(REC/'authorization.json');assert auth['approved'] and auth['lock_sha']==sha(REC/'lock.json')
    script=Path(__file__).parent/f'run_{mode}.sh'
    assert b'\r' not in script.read_bytes()
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    subprocess.run(['bash','-n',str(script)],check=True)
    t=subprocess.run(['sbatch','--test-only','--no-requeue',str(script)],env=env,text=True,capture_output=True)
    save(REC/f'{mode}_test_only.json',dict(returncode=t.returncode,stdout=t.stdout,stderr=t.stderr));assert t.returncode==0
    cmd=['sbatch','--parsable','--no-requeue',f'--output=runs/7b-{mode}-state-%j.log',f'--error=runs/7b-{mode}-state-%j.log',str(script)]
    with (REC/f'{mode}_claim.json').open('x') as f:json.dump(dict(command=cmd,authorization=auth),f)
    r=subprocess.run(cmd,env=env,text=True,capture_output=True,cwd=ROOT)
    v=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip());save(REC/f'{mode}_receipt.json',v);print(json.dumps(v))
    assert r.returncode==0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prune','selector'],required=True);main(ap.parse_args().mode)
