"""Submit the user-approved 15-minute profile once; never retry."""
import json, os, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
expected=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert ROOT.resolve()==expected.resolve()
P=ROOT/'runs/hard_mask_prepare_20261102_01'
assert json.loads((P/'CPU_cluster.json').read_text())['status']=='passed_tiny_random_weights_only'
assert json.loads((P/'CPU_media_cluster.json').read_text())['real_pairs']==44
with (P/'submission_attempt.json').open('x') as f:
    json.dump({'scope':'15 minute hard-mask profile only','approval':'User approved the proposed attempt; CPU checks and bounded profile outlined before submission',
               'GPU':1,'CPU':4,'memory_GiB':24,'minutes':15,'auto_retry':False},f)
env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
cmd=['sbatch','--parsable','--no-requeue','--partition=g078t','--comment=gx_llm',
     'experiments/2b-hard-mask-prune-v1/run_profile.sh']
result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,check=False)
receipt={'command':cmd,'returncode':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip()}
(P/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
assert result.returncode==0
