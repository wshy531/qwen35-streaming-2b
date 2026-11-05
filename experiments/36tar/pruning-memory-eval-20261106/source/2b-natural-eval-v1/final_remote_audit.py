"""Only our project, repository status, known source hashes and own Slurm IDs."""
import json,hashlib
from pathlib import Path
from cluster_io import command,LOCAL,HERE
def main():
    lock=json.loads((LOCAL/'source_lock.json').read_text(encoding='utf-8'))
    for name in ['diagnose_early9.py','temporal_route.py','evaluate_route.py']:lock[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    code='expected='+repr(lock)+r'''
import json,pathlib,subprocess,hashlib,os
p=pathlib.Path.cwd();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
if not p.resolve().is_relative_to(allowed):raise ValueError('unauthorized project')
code=p/'experiments/2b-natural-eval-v1'
sources={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in expected}
if sources!=expected:raise ValueError('final source drift')
repo=p/'repo'
def cmd(args):return subprocess.check_output(args,text=True).strip()
queue=cmd(['squeue','-u','u2025010462','--noheader','-o','%i|%j|%T|%P|%b'])
jobs=cmd(['sacct','-j','217141,217145,217146','--parsable2','--noheader','--format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES'])
result=dict(status='read_only_final_audit',physical_project=str(p.resolve()),all_21_runtime_sources_match=True,source_sha256=sources,repository_branch=cmd(['git','-C',str(repo),'branch','--show-current']),repository_commit=cmd(['git','-C',str(repo),'rev-parse','HEAD']),repository_status=cmd(['git','-C',str(repo),'status','--short']),own_queue=queue,jobs=jobs,submission_overrides={k:os.environ[k] for k in ['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS'] if k in os.environ})
print(json.dumps(result))
'''
    s=command(code,timeout=90);(LOCAL/'remote_final_audit.json').write_text(s,encoding='utf-8');x=json.loads(s)
    print(json.dumps({k:v for k,v in x.items() if k!='source_sha256'}))
if __name__=='__main__':main()
