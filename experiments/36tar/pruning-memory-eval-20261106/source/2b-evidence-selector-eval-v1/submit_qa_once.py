"""Submit exactly once only after matching explicit task-specific authorization."""
import os,subprocess,json
from selector_qa_common import ROOT,PREP,read,save,sha
from selector_qa_freeze import verify

def main():
    lock=verify();auth=read(PREP/'GPU_authorization.json')
    if (not auth.get('approved') or auth['execution_lock_sha256']!=sha(PREP/'execution_lock.json')
        or auth['budget']!=lock['budget']):raise ValueError('Exact current authorization required')
    clean={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    command=['sbatch','--parsable','--no-requeue','experiments/2b-evidence-selector-eval-v1/run_eval.sh']
    save(PREP/'gpu_submission_intent.json',dict(command=command,cwd=str(ROOT),
        removed_variable_names=sorted(set(os.environ)-set(clean)),automatic_retry=False))
    try:
        r=subprocess.run(command,cwd=ROOT,env=clean,capture_output=True,text=True,timeout=40)
        result=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    except subprocess.TimeoutExpired:
        result=dict(status='submission_unknown_do_not_retry')
    save(PREP/'gpu_submission_result.json',result);print(json.dumps(result))

if __name__=='__main__':main()
