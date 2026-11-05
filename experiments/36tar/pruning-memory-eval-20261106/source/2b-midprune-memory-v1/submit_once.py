"""One submission, after explicit approval of this pilot's concrete GPU budget."""
import json,os,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/midprune_memory_prepare_20261102_01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert str(R).startswith('/home/') and R.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['execution_lock_sha256']==sha(P/'execution_lock.json')
for n,h in read(P/'execution_lock.json').items():assert sha(R/n)==h,n
for prefix in ('CPU','CPU_layout'):
    a=read(P/f'{prefix}_local.json');b=read(P/f'{prefix}_cluster.json');assert a==b and a['passed']
media=read(P/'CPU_media_cluster.json');assert media['passed'] and len(media['read_only_media'])==5 and media['pixel_verified_pairs']==429
subprocess.run(['bash','-n','experiments/2b-midprune-memory-v1/run_pilot.sh'],cwd=R,check=True)
subprocess.run([str(R/'envs/midprune-memory-v1/bin/python'),'experiments/2b-midprune-memory-v1/check_analysis.py'],cwd=R,check=True)
cmd=['sbatch','--parsable','--no-requeue','experiments/2b-midprune-memory-v1/run_pilot.sh']
with (P/'submission_attempt.json').open('x') as f:json.dump(dict(command=cmd),f)
env=os.environ.copy();removed=[k for k in env if k.startswith('SBATCH_') or k in ('SLURM_ACCOUNT','SLURM_QOS')]
for k in removed:env.pop(k)
result=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True)
with (P/'submission_receipt.json').open('x') as f:json.dump(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,cleared_variable_names=removed),f)
print(result.stdout.strip());print(result.stderr.strip());assert result.returncode==0
