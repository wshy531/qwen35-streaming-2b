"""Single Slurm submission after task authorization and concrete resource disclosure."""
import json,os,subprocess
from pathlib import Path
from prepare import R,P,read,sha,save
assert str(R).startswith('/home/') and R.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['execution_lock_sha256']==sha(P/'execution_lock.json')
for n,h in read(P/'execution_lock.json').items():assert sha(R/n)==h,n
local=read(P/'CPU_local.json');remote=read(P/'CPU_cluster.json');media=read(P/'CPU_media_cluster.json')
assert local==remote and local['passed'] and media['passed'] and media['videos']==40
assert read(P/'CPU_analysis.json')['passed']
subprocess.run(['bash','-n','experiments/2b-prune-checkpoints-v1/run_eval.sh'],cwd=R,check=True)
with (P/'submission_attempt.json').open('x') as f:json.dump(dict(command=['sbatch','--parsable','--no-requeue','experiments/2b-prune-checkpoints-v1/run_eval.sh']),f)
env=os.environ.copy()
removed=[k for k in env if k.startswith('SBATCH_') or k in ('SLURM_ACCOUNT','SLURM_QOS')]
for k in removed:env.pop(k)
run=subprocess.run(['sbatch','--parsable','--no-requeue','experiments/2b-prune-checkpoints-v1/run_eval.sh'],cwd=R,env=env,text=True,capture_output=True)
save(P/'submission_receipt.json',dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,cleared_variable_names=removed))
print(run.stdout.strip());print(run.stderr.strip());assert run.returncode==0
