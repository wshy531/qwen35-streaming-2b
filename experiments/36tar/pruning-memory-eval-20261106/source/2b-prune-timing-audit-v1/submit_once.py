"""Submission disabled until an explicit task-specific approval is recorded."""
import os,subprocess
from audit_common import *
assert ROOT.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
a=read(PREP/'GPU_authorization.json')
assert a['approved'] and a['execution_lock_sha256']==sha(PREP/'execution_lock.json')
verify()
local=read(PREP/'CPU_local_v2.json');remote=read(PREP/'CPU_cluster.json')
assert local['status']==remote['status']=='passed'
for k in ('layouts','question_layouts','rows','identical_ID_checks','old_session_exact_pairs','tiny_QA'):
    assert local[k]==remote[k],k
assert read(PREP/'CPU_media_cluster.json')['replayed_pairs']==513
subprocess.run(['bash','-n','experiments/2b-prune-timing-audit-v1/run_audit.sh'],cwd=ROOT,check=True)
cmd=['sbatch','--parsable','--no-requeue','experiments/2b-prune-timing-audit-v1/run_audit.sh']
save(PREP/'submission_attempt.json',dict(command=cmd))
env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
r=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
save(PREP/'submission_receipt.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
print(r.stdout.strip());print(r.stderr.strip());assert r.returncode==0
