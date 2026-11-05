"""Read only the single approved job and bounded project logs, never allocate."""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import getpass,json,subprocess,time
from pathlib import Path
p=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
assert getpass.getuser()=='u2025010462' and Path.cwd().resolve()==p.resolve()
assert p.resolve().is_relative_to(Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
job='217135'
def call(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=15)
 return dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
def small(path,limit=131072):
 if not path.is_file():return None
 if path.stat().st_size>limit:return dict(bytes=path.stat().st_size,too_large_to_read=True)
 return dict(bytes=path.stat().st_size,text=path.read_text(errors='replace'))
def tail(path):
 if not path.is_file():return None
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-16384));data=f.read(16384)
 return dict(bytes=path.stat().st_size,tail=data.decode(errors='replace'))
folder=p/('runs/first-2b-gpu-'+job)
result=dict(status='read_only_job_observation',job_id=job,observed_epoch=time.time(),
 queue=call(['squeue','-u','u2025010462','-j',job,'--noheader','-o','%i|%j|%T|%P|%M|%l|%D|%C|%b|%R']),
 accounting=call(['sacct','-j',job,'--noheader','--parsable2','--format=JobIDRaw,State,ExitCode,Elapsed,AllocTRES,NodeList']),
 control=call(['scontrol','show','job',job,'-o']),project_output_exists=folder.is_dir(),files={})
for name in ['job_id.txt','hostname.txt','original/resource_guard.json','original/summary.json','stream/resource_guard.json','stream/summary.json','comparison/comparison.json']:
 result['files'][name]=small(folder/name)
for name in ['original.log','stream.log','comparison.log']:result['files'][name]=tail(folder/name)
for name in ['out','err']:result['files']['slurm.'+name]=tail(p/('logs/first-2b-'+job+'.'+name))
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=Path(a.run_dir).resolve()
    assert out.parent==(root/'runs').resolve() and out.is_dir()
    assert json.loads((out/'submission_result.json').read_text())['job_id']=='217135'
    i=1
    while (out/f'observation_{i:03d}.json').exists():i+=1
    project='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
    cmd=['C:/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
        '-o','ConnectTimeout=15','-i',str(Path.home()/'.ssh/qwen35_cluster_temp'),'ChaoSuan',
        'cd '+project+' && PYTHONDONTWRITEBYTECODE=1 '+project+'/envs/qwen35-2b-fp32/bin/python -']
    proc=subprocess.run(cmd,input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=80)
    if proc.returncode:
        value=dict(status='observation_connection_failed',exit=proc.returncode,stderr=proc.stderr)
    else:value=json.loads(proc.stdout)
    (out/f'observation_{i:03d}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k not in ['files','control']},ensure_ascii=False))
    for name,x in value.get('files',{}).items():
        if x:
            if name.endswith('summary.json') or name.endswith('comparison.json'):
                d=json.loads(x['text']);print(name,json.dumps({k:d[k] for k in ['status','stage','implementation','pretrained_weights_loaded','execution','passed'] if k in d},ensure_ascii=False), 'records',len(d.get('records',d.get('cases',[]))))
            elif name.endswith('.log') or name=='slurm.err':print(name,x.get('tail','')[-1600:])


if __name__=='__main__':main()
