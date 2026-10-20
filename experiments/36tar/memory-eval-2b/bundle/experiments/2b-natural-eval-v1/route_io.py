"""Scoped R0 preparation, CPU preflight, one authorized GPU job and receipts."""
import argparse,base64,hashlib,json,shlex
from datetime import datetime
from pathlib import Path
from cluster_io import command,LOCAL,HERE,PROJECT
ROOT=HERE.parents[1];OUT=ROOT/'runs/2b_temporal_route_20261020_01';RR='runs/2b-temporal-route-inputs-20261020-01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    from temporal_route import TEMPORAL
    OUT.mkdir(exist_ok=False);(OUT/'requests.json').write_bytes((LOCAL/'requests.json').read_bytes());rs=read(OUT/'requests.json')
    triggers={c:sum(bool(TEMPORAL.search(r['model_visible']['question'])) for r in rs if r['cohort']==c) for c in ['public74','open47','new50']}
    save(OUT/'protocol.json',dict(status='single_fixed_candidate_before_new_outputs',candidate='R0_lexical_temporal_gate',questions=171,videos=67,new_sources={n:sha(HERE/n) for n in ['temporal_route.py','evaluate_route.py']},base_source_lock_sha256=sha(LOCAL/'source_lock.json'),requests_sha256=sha(LOCAL/'requests.json'),arms=['all_memory','long_only','recent4','temporal_route'],storage='unchanged M2 forest384 with fixed threshold, no new event representation',read_caps=dict(all_memory=384,long_only=192,recent4=192,temporal_route=384),fallback='if no long nodes, read all available bank',regex=TEMPORAL.pattern,temporal_cue_counts_before_outputs=triggers,inputs='original question only for routing; final original full question/options unchanged',maximum_answers=684,maximum_generated_tokens=5472,maximum_real_model_loads=1,CPU_context_checks=684,worker_seconds=1200,deadline_UTC='2026-10-20T06:11:50+00:00',precision='float32',resources=dict(GPU=1,CPU=4,RAM_GiB=24,Slurm_minutes=25,comment='gx_llm',partition='g078t',requeue=False),analysis='per-cohort and F/T primary; strict and leading-format separate; pairs vs all_memory/recent4/M0; cluster bootstrap; all_memory full-ID replay; no prompt/regex/threshold tuning',data_status='171 already-open comparison questions, not independent final holdout',stop='one candidate round, no second routing rule or rescue prompt in this window',limitations='Lexical cue is not semantic event localization. Before/after may require recent evidence. Reduced read budget is explicit, not equal actual Token count.'))
    print(json.dumps(dict(questions=171,answers=684,cue_counts=triggers)))
def install_cpu():
    payload={f'experiments/2b-natural-eval-v1/{n}':base64.b64encode((HERE/n).read_bytes()).decode() for n in ['temporal_route.py','evaluate_route.py']}
    payload[RR+'/protocol.json']=base64.b64encode((OUT/'protocol.json').read_bytes()).decode()
    code='payload='+repr(payload)+r'''
import json,pathlib,subprocess,os,time,base64,hashlib
p=pathlib.Path.cwd();run=p/'runs/2b-temporal-route-inputs-20261020-01';files=[]
for rel,b64 in payload.items():
 dst=p/rel
 if not dst.resolve().is_relative_to(p.resolve()):raise ValueError('outside project')
 dst.parent.mkdir(parents=True,exist_ok=True);value=base64.b64decode(b64)
 with dst.open('xb') as f:f.write(value)
 files.append(dict(path=rel,sha256=hashlib.sha256(value).hexdigest(),bytes=len(value)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
start=time.monotonic()
with (run/'cpu.log').open('x') as log:r=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(p/'experiments/2b-natural-eval-v1/evaluate_route.py'),'cpu','--run',str(run)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
print(json.dumps(dict(exit_code=r.returncode,seconds=time.monotonic()-start,files=files,proof=json.loads((run/'cpu_preflight.json').read_text()) if (run/'cpu_preflight.json').exists() else None,log_tail=(run/'cpu.log').read_text()[-3500:])))
'''
    s=command(code,timeout=300);(OUT/'install_cpu.json').write_text(s,encoding='utf-8');r=json.loads(s);print(json.dumps(dict(exit_code=r['exit_code'],seconds=r['seconds'],tail=r['log_tail'])))
def submit():
    if read(OUT/'install_cpu.json')['exit_code']!=0:raise ValueError('CPU proof required')
    with (OUT/'submission_attempt.json').open('x') as f:json.dump(dict(attempt=1,authorization='user two-hour measurement GPU window'),f)
    lines=['set -eu','cd '+shlex.quote(PROJECT),'test -n "${SLURM_JOB_ID:-}"','test -n "${CUDA_VISIBLE_DEVICES:-}"','export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false']
    for k,v in [('HF_HOME','cache/huggingface'),('XDG_CACHE_HOME','cache/xdg'),('TORCH_HOME','cache/torch'),('TORCHINDUCTOR_CACHE_DIR','cache/torchinductor'),('TRITON_CACHE_DIR','cache/triton'),('TMPDIR','cache/tmp'),('CUDA_CACHE_PATH','cache/cuda')]:lines.append('export '+k+'='+shlex.quote(PROJECT+'/'+v))
    lines+=['seconds=$((1792476710-$(date -u +%s))); test "$seconds" -gt 60; if [ "$seconds" -gt 1200 ]; then seconds=1200; fi','exec timeout "$seconds" '+shlex.quote(PROJECT+'/envs/qwen35-2b-fp32/bin/python')+' '+shlex.quote(PROJECT+'/experiments/2b-natural-eval-v1/evaluate_route.py')+' worker --run '+shlex.quote(PROJECT+'/'+RR)+' --out '+shlex.quote(PROJECT+'/runs/2b-temporal-route-')+'${SLURM_JOB_ID}']
    wrapper='\n'.join(lines)+'\n';(OUT/'job_wrapper.sh').write_text(wrapper,encoding='utf-8')
    code='wrapper='+repr(wrapper)+r'''
import pathlib,json,subprocess,os,time,hashlib
p=pathlib.Path.cwd();run=p/'runs/2b-temporal-route-inputs-20261020-01';proof=json.loads((run/'cpu_preflight.json').read_text())
if proof['status']!='passed' or proof['protocol_sha256']!=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest():raise ValueError('CPU identity')
if subprocess.check_output(['squeue','-u','u2025010462','--noheader'],text=True).strip():raise ValueError('own queue not empty')
if any(k in os.environ for k in ['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS']):raise ValueError('scheduler overrides')
if 1792476710-time.time()<1500:raise ValueError('less than25 minutes remaining')
subprocess.run(['sh','-n'],input=wrapper,text=True,check=True)
args=['sbatch','--parsable','--job-name=q35-2b-route-r0','--partition=g078t','--comment=gx_llm','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=24G','--time=00:25:00','--no-requeue','--chdir='+str(p),'--output='+str(run/'slurm-%j.out'),'--error='+str(run/'slurm-%j.err'),'--wrap',wrapper]
with (run/'submission_attempt.json').open('x') as f:json.dump(dict(arguments=args),f,indent=2)
r=subprocess.run(args,text=True,capture_output=True,timeout=30);result=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,arguments=args)
if r.returncode==0:result['job_id']=r.stdout.strip().split(';')[0];result['allocation']=subprocess.check_output(['scontrol','show','job','-dd',result['job_id']],text=True)
(run/'submission_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    s=command(code);(OUT/'submission_result.json').write_text(s,encoding='utf-8');print(s)
def observe():
    job=read(OUT/'submission_result.json')['job_id'];code='job='+repr(job)+r'''
import pathlib,subprocess,json
p=pathlib.Path.cwd();run=p/'runs/2b-temporal-route-inputs-20261020-01';out=p/('runs/2b-temporal-route-'+job);pred=out/'predictions.jsonl'
print(json.dumps(dict(job=job,queue=subprocess.check_output(['squeue','-j',job,'--noheader','-o','%i|%T|%M|%R'],text=True),accounting=subprocess.check_output(['sacct','-j',job,'--parsable2','--noheader','--format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES'],text=True),predictions=sum(1 for _ in pred.open()) if pred.exists() else 0,completed=json.loads((out/'completed.json').read_text()) if (out/'completed.json').exists() else None,stderr=(run/('slurm-'+job+'.err')).read_text()[-2500:] if (run/('slurm-'+job+'.err')).exists() else '')))
'''
    s=command(code);(OUT/('observation_'+datetime.now().strftime('%H%M%S')+'.json')).write_text(s,encoding='utf-8');print(s)
def fetch():
    job=read(OUT/'submission_result.json')['job_id'];code='job='+repr(job)+r'''
import pathlib,json,base64,hashlib
p=pathlib.Path.cwd();files=[];total=0
for folder,prefix in [(p/('runs/2b-temporal-route-'+job),'result'),(p/'runs/2b-temporal-route-inputs-20261020-01','input')]:
 for src in folder.rglob('*'):
  if not src.is_file():continue
  value=src.read_bytes();total+=len(value)
  if len(value)>16*2**20 or total>32*2**20:raise ValueError('bounded evidence')
  files.append(dict(path=prefix+'/'+src.relative_to(folder).as_posix(),sha256=hashlib.sha256(value).hexdigest(),base64=base64.b64encode(value).decode()))
print(json.dumps(files))
'''
    rs=json.loads(command(code,timeout=180));dest=OUT/('gpu_'+job);dest.mkdir(exist_ok=False)
    for r in rs:
        value=base64.b64decode(r.pop('base64'));p=dest/r['path']
        if not p.resolve().is_relative_to(dest.resolve()) or hashlib.sha256(value).hexdigest()!=r['sha256']:raise ValueError('evidence identity')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
    save(OUT/'fetch_receipt.json',rs);print(json.dumps(dict(files=len(rs),output=str(dest))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','install_cpu','submit','observe','fetch']);a=p.parse_args();globals()[a.mode]()
