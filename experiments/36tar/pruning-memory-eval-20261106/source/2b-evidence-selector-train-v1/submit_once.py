"""One feature-stage submission, exact authorization and exclusive intent receipt."""
import os,subprocess,json
from prepare_data import ROOT,PREP,read,save,sha
from selector_freeze import verify

def main():
    lock=verify();a=read(PREP/'GPU_authorization.json')
    if not a['approved'] or a['budget']!=lock['budget'] or a['execution_lock_sha256']!=sha(PREP/'execution_lock.json'):
        raise ValueError('current exact authorization required')
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    command=['sbatch','--parsable','--no-requeue','experiments/2b-evidence-selector-train-v1/run_features.sh']
    save(PREP/'gpu_submission_intent.json',dict(command=command,cwd=str(ROOT),automatic_retry=False))
    try:
        r=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=40)
        receipt=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    except subprocess.TimeoutExpired:receipt=dict(status='unknown_submission_timeout_do_not_retry')
    save(PREP/'gpu_submission_result.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
