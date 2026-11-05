"""Single read-only SSH preflight for a separately authorized original retry."""
import json
from pathlib import Path
import subprocess


REMOTE=r'''
import getpass,hashlib,json,os,socket,subprocess
from pathlib import Path
project=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
root=Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
if not project.resolve().is_relative_to(root) or Path.cwd().resolve()!=project.resolve():raise ValueError('unauthorized working path')
if getpass.getuser()!='u2025010462':raise ValueError('wrong remote user')
repo=project/'repo';active=repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
code=repo/'experiments/36tar/2b-memory-eval';proof=project/'runs/first-2b-preflight-20261016-02'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def command(args):return subprocess.check_output(args,text=True).strip()
branch=command(['git','-C',str(repo),'branch','--show-current'])
queue=command(['squeue','-u','u2025010462','--noheader','-o','%i|%j|%T|%P|%b'])
names=['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS']
overrides={k:os.environ[k] for k in names if k in os.environ}
runfile=code/'run_first_2b_baseline.sh'
syntax=subprocess.run(['sh','-n',str(runfile)],capture_output=True,text=True)
proofs=[]
for impl,source in [('original',repo/'qwen35_local/modeling_qwen3_5.py'),('stream',active/'qwen35_local/modeling_qwen3_5.py')]:
 saved=json.loads((proof/impl/'summary.json').read_text())
 identity=dict(script_sha256=sha(code/'first_2b_baseline.py'),model_manifest_sha256=sha(project/'models/Qwen3.5-2B/DOWNLOAD_MANIFEST.json'),model_source_sha256=sha(source),online_source_sha256=sha(active/'streaming/online.py'),hybrid_source_sha256=sha(active/'streaming/hybrid.py'),question='Describe the visible scene and activity briefly.',canvas=[192,256],source_sha256={s:sha(project/'data/ovo-pilot-dev4'/s) for s in saved['identity']['source_sha256']},max_new_tokens=16)
 proofs.append(dict(implementation=impl,stage=saved['stage'],status=saved['status'],records=len(saved['records']),identity_matches=saved['identity']==identity,identity=identity))
manifest=json.loads((project/'models/Qwen3.5-2B/DOWNLOAD_MANIFEST.json').read_text())
asset_sizes=[]
for f in manifest['files']:
 path=project/'models/Qwen3.5-2B'/f['name']
 asset_sizes.append(dict(name=f['name'],size_matches=path.is_file() and path.stat().st_size==f['bytes']))
runsha=sha(runfile)
result=dict(status='read_only_no_submission',user=getpass.getuser(),hostname=socket.gethostname(),project=str(project),physical_project=str(project.resolve()),branch=branch,commit=command(['git','-C',str(repo),'rev-parse','HEAD']),git_status=command(['git','-C',str(repo),'status','--short']),own_queue=queue,submission_override_env=overrides,run_script_sha256=runsha,run_script_shell_syntax_ok=syntax.returncode==0,proofs=proofs,model_asset_sizes=asset_sizes,inside_allocation=bool(os.environ.get('SLURM_JOB_ID')))
result['preflight_ok']=(branch=='36tar/2b-memory-eval' and not queue and not overrides and not result['inside_allocation'] and syntax.returncode==0 and runsha=='21731748f40182c80ebba78e4b1ac7ba4942d5590f60c3c65882cd93cd1a91c8' and all(p['stage']=='preflight' and p['status']=='complete' and p['identity_matches'] for p in proofs) and all(a['size_matches'] for a in asset_sizes))
print(json.dumps(result))
'''


def main():
    root=Path(__file__).resolve().parents[2]
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    out=Path(parser.parse_args().run_dir).resolve()
    if out.parent!=(root/'runs').resolve():raise ValueError('run directory must be directly under project runs')
    out.mkdir(exist_ok=False)
    (out/'readonly_preflight_remote.py').write_text(REMOTE,encoding='utf-8')
    project='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
    client=Path('C:/Windows/System32/OpenSSH/ssh.exe')
    args=[str(client),'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o','ConnectTimeout=15','-i',str(Path.home()/'.ssh/qwen35_cluster_temp'),'ChaoSuan',
        'cd '+project+' && PYTHONDONTWRITEBYTECODE=1 '+project+'/envs/qwen35-2b-fp32/bin/python -']
    try:
        p=subprocess.run(args,input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=90)
    except subprocess.TimeoutExpired:
        result=dict(status='SSH_preflight_timed_out',submissions=0,preflight_ok=False)
        (out/'preflight_connection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result));return
    (out/'preflight_stdout.json').write_text(p.stdout,encoding='utf-8')
    (out/'preflight_stderr.txt').write_text(p.stderr,encoding='utf-8')
    result=dict(status='read_only_attempt_complete',exit_code=p.returncode,submissions=0,
        preflight_ok=p.returncode==0 and json.loads(p.stdout).get('preflight_ok',False))
    (out/'preflight_connection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    if p.returncode:print(p.stderr[:1200])
    else:print(p.stdout)


if __name__=='__main__':main()
