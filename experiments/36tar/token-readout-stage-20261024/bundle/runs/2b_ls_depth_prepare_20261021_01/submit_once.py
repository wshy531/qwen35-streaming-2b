"""Run only after explicit approval of this concrete GPU task. No retries."""
import datetime
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/2b-ls-depth-v1'))
from common import read, sha, save
from evaluate import verify
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import command

LOCK = '600ef2ed8108863d5192aa05ec3bfd6840839f5aab0682a333d91554183ef08a'


def main():
    inp = OUT/'inputs'
    assert sha(inp/'execution_lock.json') == LOCK
    protocol, rows, groups = verify(inp)
    assert read(OUT/'CPU_preflight.json')['status'] == 'passed'
    assert read(OUT/'cluster_CPU_execution.json')['exit_code'] == 0
    assert read(OUT/'final_cluster_checks.json')['status'] == 'passed'
    analysis = read(OUT/'analysis_lock.json')
    for key,path in [('labels_sha256',OUT/'labels_LOCAL_ONLY.json'),
                     ('scorer_sha256',ROOT/'experiments/2b-ls-depth-v1/analyze.py'),
                     ('historical_references_sha256',OUT/'historical_references_LOCAL_ONLY.json'),
                     ('scorer_contract_sha256',OUT/'scorer_contract.json')]:
        assert sha(path) == analysis[key]
    # This receipt is a human-supplied gate, never produced by prepare/deploy.
    approval = read(OUT/'specific_GPU_user_approval.json')
    assert approval['approved'] is True and approval['execution_lock_sha256'] == LOCK
    auth = dict(approved=True,execution_lock_sha256=LOCK,protocol_sha256=sha(inp/'protocol.json'),
        user_approval=approval,resources=protocol['resources'],arms=protocol['arms'],
        questions=303,answers_max=1818,generated_tokens_max=14544,vision_encodings=3981,
        no_auto_retry=True,no_requeue=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(OUT/'local_submit_attempt.json',dict(execution_lock_sha256=LOCK,authorization=auth))
    code = """import datetime,getpass,hashlib,json,os,pathlib,subprocess
p=pathlib.Path.cwd().resolve();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert p.is_relative_to(allowed) and p.name=='qwen35-streaming-2b-36tar'
run=p/'runs/2b_ls_depth_prepare_20261021_01';inp=run/'inputs';auth=AUTH
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(inp/'execution_lock.json')==auth['execution_lock_sha256']
lock=json.loads((inp/'execution_lock.json').read_text())
for rel,h in lock['runtime_sha256'].items():assert sha(p/rel)==h,rel
for rel,h in lock['input_sha256'].items():assert sha(inp/rel)==h,rel
cpu=json.loads((run/'CPU_preflight.json').read_text());assert cpu['status']=='passed' and cpu['protocol_sha256']==auth['protocol_sha256']
keys=['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS']
overrides={k:os.environ[k] for k in keys if os.environ.get(k)};assert not overrides,overrides
queue=subprocess.run(['squeue','-u',getpass.getuser(),'-h','-o','%i %T %j'],capture_output=True,text=True,check=True).stdout
assert not queue.strip(),'existing user jobs, review first'
script=p/'experiments/2b-ls-depth-v1/run_eval.sh';submitted=script.read_bytes().replace(bytes([13,10]),bytes([10])).decode()
lines=[line for line in submitted.splitlines() if line.startswith('#SBATCH')]
assert lines==['#SBATCH --job-name=2b-ls-depth','#SBATCH --partition=g078t','#SBATCH --comment=gx_llm','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=4','#SBATCH --gres=gpu:1','#SBATCH --mem=24G','#SBATCH --time=01:00:00']
subprocess.run(['bash','-n'],input=submitted,capture_output=True,text=True,check=True)
args=['sbatch','--parsable','--no-requeue','--output='+str(run/'slurm-%j.log'),'--error='+str(run/'slurm-%j.err')]
attempt={'command':args,'stdin_script':str(script),'script_sha256':sha(script),
 'submitted_script_sha256':hashlib.sha256(submitted.encode()).hexdigest(),'SBATCH_lines':lines,
 'environment_overrides':overrides,'queue_before':queue,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_auto_retry':True}
assert not (run/'submission_attempt.json').exists()
with (inp/'GPU_authorization.json').open('x') as stream:json.dump(auth,stream,indent=2)
with (run/'submission_attempt.json').open('x') as stream:json.dump(attempt,stream,indent=2)
result=subprocess.run(args,input=submitted,capture_output=True,text=True,timeout=30)
receipt={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'attempt':attempt}
if result.returncode==0:
 job=result.stdout.strip().split(';')[0];assert job.isdigit();receipt['job_id']=job
 control=subprocess.run(['scontrol','show','job','-dd',job],capture_output=True,text=True,timeout=15)
 receipt['scontrol']=control.stdout;receipt['scontrol_stderr']=control.stderr
with (run/'submission_receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
""".replace('AUTH',repr(auth))
    result = command(code,timeout=100)
    with (OUT/'submission_receipt.json').open('x',encoding='utf-8') as stream: stream.write(result)
    print(result)


if __name__ == '__main__': main()
