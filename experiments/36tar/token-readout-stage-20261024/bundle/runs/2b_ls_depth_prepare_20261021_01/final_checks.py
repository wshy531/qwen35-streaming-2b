"""Final read-only source and scheduler audit; no submission or pretrained load."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import command


def main():
    code = """import getpass,hashlib,json,os,pathlib,subprocess
p=pathlib.Path.cwd().resolve();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert p.is_relative_to(allowed) and p.name=='qwen35-streaming-2b-36tar'
run=p/'runs/2b_ls_depth_prepare_20261021_01';inp=run/'inputs'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
lock=json.loads((inp/'execution_lock.json').read_text())
assert sha(inp/'execution_lock.json')=='600ef2ed8108863d5192aa05ec3bfd6840839f5aab0682a333d91554183ef08a'
for rel,h in lock['runtime_sha256'].items():assert sha(p/rel)==h,rel
for rel,h in lock['input_sha256'].items():assert sha(inp/rel)==h,rel
cpu=json.loads((run/'CPU_preflight.json').read_text());assert cpu['status']=='passed' and cpu['protocol_sha256']==lock['protocol_sha256']
assert sha(run/'CPU_local_receipt.json')==lock['CPU_receipt_sha256']
keys=['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS']
overrides={k:os.environ[k] for k in keys if os.environ.get(k)}
assert not overrides,overrides
script=p/'experiments/2b-ls-depth-v1/run_eval.sh'
subprocess.run(['bash','-n',str(script)],capture_output=True,text=True,check=True)
queue=subprocess.run(['squeue','-u',getpass.getuser(),'-h','-o','%i %T %j'],capture_output=True,text=True,check=True).stdout
groups=json.loads((inp/'groups.json').read_text());media=set()
for g in groups:
 path=p/g['cluster_media'];assert path.resolve().is_relative_to(p) and path.is_file();media.add(str(path))
assert not (inp/'GPU_authorization.json').exists()
print(json.dumps({'status':'passed','project':str(p),'user':getpass.getuser(),'queue':queue,'scheduler_overrides':overrides,
 'runtime_sources':len(lock['runtime_sha256']),'CPU':{k:cpu[k] for k in ['status','seconds','context_layouts','pretrained_loads','GPU_submissions']},
 'media_files_exist':len(media),'GPU_authorization_present':False,'GPU_submissions':0,
 'SBATCH_lines':[line for line in script.read_text().splitlines() if line.startswith('#SBATCH')],
 'disk':subprocess.run(['df','-h',str(p)],capture_output=True,text=True,check=True).stdout}))
"""
    result=command(code,timeout=90)
    with (OUT/'final_cluster_checks.json').open('x',encoding='utf-8') as stream:stream.write(result)
    print(result)


if __name__ == '__main__':main()
