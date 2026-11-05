"""Requires task-specific GPU approval bound to this execution lock."""
import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/recent_detail_prepare_20261103_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert R.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['execution_lock_sha256']==sha(P/'execution_lock.json')
for n,h in read(P/'execution_lock.json').items():
    p=R/n
    if not p.exists() and n.startswith('repo/'):p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
for stem in ('CPU','CPU_QA','CPU_layout'):
    assert read(P/f'{stem}_local.json')==read(P/f'{stem}_cluster.json') and read(P/f'{stem}_local.json')['passed']
assert read(P/'CPU_media_cluster.json')['pixel_verified_pairs']==429
subprocess.run(['bash','-n','experiments/2b-recent-detail-baseline-v1/run_pilot.sh'],cwd=R,check=True)
cmd=['sbatch','--parsable','--no-requeue','experiments/2b-recent-detail-baseline-v1/run_pilot.sh']
with (P/'submission_attempt.json').open('x') as f:json.dump(dict(command=cmd),f)
env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
r=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True)
with (P/'submission_receipt.json').open('x') as f:json.dump(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),f)
print(r.stdout.strip());print(r.stderr.strip());assert r.returncode==0
