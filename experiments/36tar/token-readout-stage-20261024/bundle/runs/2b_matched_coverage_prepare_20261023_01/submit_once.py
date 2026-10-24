"""Single explicitly authorized sbatch, with immutable prechecks and receipt."""
from pathlib import Path
import hashlib,json,os,subprocess,time
ROOT=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar').resolve()
assert ROOT.is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
OUT=ROOT/'runs/2b_matched_coverage_prepare_20261023_01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
lock=read(OUT/'inputs/execution_lock.json')
assert sha(OUT/'inputs/execution_lock.json')=='4e569fa6b45e803b21fbfe6fec13f164b14c04fa437731f8f38daf59870f81b8'
auth=read(OUT/'inputs/GPU_authorization.json')
assert auth['approved'] and auth['execution_lock_sha256']==sha(OUT/'inputs/execution_lock.json')
assert sha(OUT/'CPU_preflight.json')=='cb6c7c6d5959a37aa9b1baeb3ace2c982f2819f7c46c22f0dc69ad9a7a00684a'
assert sha(OUT/'tiny_session_check.json')=='4a3803d06dfc3dbb8f107f89eda729818af41fb9454a1b89a1e7e9ac4e7fc72d'
for rel,digest in lock['runtime_sha256'].items():
    p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,rel
for name,digest in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==digest,name
variables=('SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS')
assert not {k:os.environ[k] for k in variables if os.environ.get(k)}
queue=subprocess.run(['squeue','-u','u2025010462','-h','-o','%i %j %T'],capture_output=True,text=True,check=True)
assert not queue.stdout.strip(),queue.stdout
script=ROOT/'experiments/2b-matched-coverage-v1/run_eval.sh'
assert b'\r' not in script.read_bytes()
command=['sbatch','--parsable','--no-requeue',
         '--output='+str(OUT/'slurm-%j.out'),'--error='+str(OUT/'slurm-%j.err'),str(script)]
save(OUT/'submission_intent.json',dict(command=command,script_sha256=sha(script),unix_time=time.time(),
    execution_lock_sha256=sha(OUT/'inputs/execution_lock.json'),no_auto_retry=True))
result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,command=command,
             script_sha256=sha(script),execution_lock_sha256=sha(OUT/'inputs/execution_lock.json'))
if result.returncode==0:
    job=result.stdout.strip().split(';')[0];assert job.isdigit();receipt['job_id']=job
save(OUT/'submission_receipt.json',receipt)
print(json.dumps(receipt))
assert result.returncode==0
