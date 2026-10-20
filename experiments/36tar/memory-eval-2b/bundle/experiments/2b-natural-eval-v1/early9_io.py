"""Fixed9 diagnostic transport and one authorized Slurm submission."""
import argparse,base64,hashlib,json,shlex,sys
from pathlib import Path
from datetime import datetime
from cluster_io import command,LOCAL,HERE,PROJECT
ROOT=HERE.parents[1];OUT=ROOT/'runs/2b_early9_diagnostic_20261020_01'
RR='runs/2b-early9-inputs-20261020-01'
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    firsts={'nextgqa:val:4838113006:7':18,'nextgqa:val:2868183184:5':1,'nextgqa:val:6618792571:6':16,'nextgqa:val:6618792571:8':25,'nextgqa:val:10030609934:8':15,'nextgqa:val:10030609934:1':3,'nextgqa:val:10030609934:2':8,'nextgqa:val:2435633172:2':21,'nextgqa:val:4013751996:5':0}
    data=ROOT/'runs/early25_full_prefix_review_20261020_01';reviews=[json.loads(s) for s in (data/'source_reviews.jsonl').read_text(encoding='utf-8').splitlines()]
    eligible={q['id'] for r in reviews for q in r['questions'] if q['content_verdict']=='supported' and q['memory_character']=='past_not_repeated_in_sample'}
    if set(firsts)!=eligible:raise ValueError('all preannotated9 required')
    original={r['id']:r for r in json.loads((LOCAL/'requests.json').read_text(encoding='utf-8'))}
    OUT.mkdir(exist_ok=False);rs=[dict(id=k,old_first_pair=v,request=original[k]) for k,v in firsts.items()];save(OUT/'requests.json',rs)
    save(OUT/'protocol.json',dict(status='posthoc_targeted_diagnostic_fixed_before_its_answers',questions=9,videos=6,request_sha256=sha(OUT/'requests.json'),driver_sha256=sha(HERE/'diagnose_early9.py'),base_source_lock_sha256=sha(LOCAL/'source_lock.json'),curation_lock_sha256=sha(data/'REVIEW_LOCK.json'),arms=['old_full','old_merged','old_full_recent4','old_merged_recent4','recent4','text_only','full_replay'],old_raw_pairs=4,old_raw_visual_tokens=192,old_merged_visual_tokens=48,recent_pairs=4,maximum_answers=63,maximum_generated_tokens=504,maximum_real_loads=1,worker_seconds=600,deadline_UTC='2026-10-20T06:11:50+00:00',precision='float32',canvas=[192,256],max_new_tokens=8,manual_old_window='fixed from full-prefix visual witnesses; not automatic retrieval or a deployable selector',compression='existing ToMe within=True and compressed position/label heuristic; not feature-value-only attribution',analysis='All9 including base errors; A/B compression-layout effect; A/C and B/D recent interference; recent4 removal control; full/text replay identity; no prompt/window sweeps',selection_bias_note='All9 content-qualified before opening2B scores; this diagnostic designed after base scores, not held-out evaluation',stop='one fixed real-model round, retain every failure'))
    print(json.dumps(dict(questions=9,answers=63,loads=1)))
def install_cpu():
    payload={'experiments/2b-natural-eval-v1/diagnose_early9.py':base64.b64encode((HERE/'diagnose_early9.py').read_bytes()).decode()}
    for name in ['requests.json','protocol.json']:payload[RR+'/'+name]=base64.b64encode((OUT/name).read_bytes()).decode()
    code='payload='+repr(payload)+r'''
import pathlib,json,base64,hashlib,subprocess,os,time
p=pathlib.Path.cwd();run=p/'runs/2b-early9-inputs-20261020-01';receipts=[]
for rel,b64 in payload.items():
 dst=p/rel
 if not dst.resolve().is_relative_to(p.resolve()):raise ValueError('project-only install')
 dst.parent.mkdir(parents=True,exist_ok=True);value=base64.b64decode(b64)
 with dst.open('xb') as f:f.write(value)
 receipts.append(dict(path=rel,sha256=hashlib.sha256(value).hexdigest(),bytes=len(value)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(p/'cache/huggingface'),TMPDIR=str(p/'cache/tmp'))
start=time.monotonic()
with (run/'cpu.log').open('x') as f:r=subprocess.run([str(p/'envs/qwen35-2b-fp32/bin/python'),str(p/'experiments/2b-natural-eval-v1/diagnose_early9.py'),'cpu','--run',str(run)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
print(json.dumps(dict(files=receipts,exit_code=r.returncode,seconds=time.monotonic()-start,proof=json.loads((run/'cpu_preflight.json').read_text()) if (run/'cpu_preflight.json').exists() else None,log_tail=(run/'cpu.log').read_text()[-3500:])))
'''
    s=command(code,timeout=300);(OUT/'install_cpu.json').write_text(s,encoding='utf-8');r=json.loads(s);print(json.dumps(dict(exit_code=r['exit_code'],seconds=r['seconds'],log_tail=r['log_tail'])))
def submit():
    if json.loads((OUT/'install_cpu.json').read_text())['exit_code']!=0:raise ValueError('CPU not passed')
    with (OUT/'submission_attempt.json').open('x') as f:json.dump(dict(attempts=1,authorization='user two-hour evaluation GPU window'),f)
    p=PROJECT
    lines=['set -eu','cd '+shlex.quote(p),'test -n "${SLURM_JOB_ID:-}"','test -n "${CUDA_VISIBLE_DEVICES:-}"','export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false']
    for key,rel in [('HF_HOME','cache/huggingface'),('XDG_CACHE_HOME','cache/xdg'),('TORCH_HOME','cache/torch'),('TORCHINDUCTOR_CACHE_DIR','cache/torchinductor'),('TRITON_CACHE_DIR','cache/triton'),('TMPDIR','cache/tmp'),('CUDA_CACHE_PATH','cache/cuda')]:lines.append('export '+key+'='+shlex.quote(p+'/'+rel))
    lines+=['seconds=$((1792476710-$(date -u +%s))); test "$seconds" -gt 60; if [ "$seconds" -gt 600 ]; then seconds=600; fi','exec timeout "$seconds" '+shlex.quote(p+'/envs/qwen35-2b-fp32/bin/python')+' '+shlex.quote(p+'/experiments/2b-natural-eval-v1/diagnose_early9.py')+' worker --run '+shlex.quote(p+'/'+RR)+' --out '+shlex.quote(p+'/runs/2b-early9-')+'${SLURM_JOB_ID}']
    wrapper='\n'.join(lines)+'\n';(OUT/'job_wrapper.sh').write_text(wrapper,encoding='utf-8')
    code='wrapper='+repr(wrapper)+r'''
import datetime,json,pathlib,subprocess,os,hashlib,time
p=pathlib.Path.cwd();run=p/'runs/2b-early9-inputs-20261020-01';proof=json.loads((run/'cpu_preflight.json').read_text())
if proof['status']!='passed' or proof['protocol_sha256']!=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest():raise ValueError('CPU proof')
if subprocess.check_output(['squeue','-u','u2025010462','--noheader'],text=True).strip():raise ValueError('own allocation still present')
if any(k in os.environ for k in ['SBATCH_ACCOUNT','SBATCH_PARTITION','SBATCH_QOS','SBATCH_COMMENT','SLURM_ACCOUNT','SLURM_QOS']):raise ValueError('scheduler environment override')
deadline=int(datetime.datetime(2026,10,20,6,11,50,tzinfo=datetime.timezone.utc).timestamp())
if deadline-time.time()<1200:raise ValueError('insufficient authorized window')
subprocess.run(['sh','-n'],input=wrapper,text=True,check=True)
args=['sbatch','--parsable','--job-name=q35-2b-early9','--partition=g078t','--comment=gx_llm','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=24G','--time=00:20:00','--no-requeue','--chdir='+str(p),'--output='+str(run/'slurm-%j.out'),'--error='+str(run/'slurm-%j.err'),'--wrap',wrapper]
with (run/'submission_attempt.json').open('x') as f:json.dump(dict(arguments=args),f,indent=2)
r=subprocess.run(args,capture_output=True,text=True,timeout=30);result=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,arguments=args)
if r.returncode==0:result['job_id']=r.stdout.strip().split(';')[0];result['allocation']=subprocess.check_output(['scontrol','show','job','-dd',result['job_id']],text=True)
(run/'submission_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
    s=command(code);(OUT/'submission_result.json').write_text(s,encoding='utf-8');print(s)
def observe():
    job=json.loads((OUT/'submission_result.json').read_text())['job_id'];code='job='+repr(job)+r'''
import json,pathlib,subprocess
p=pathlib.Path.cwd();out=p/('runs/2b-early9-'+job);run=p/'runs/2b-early9-inputs-20261020-01';pred=out/'predictions.jsonl'
print(json.dumps(dict(job=job,queue=subprocess.check_output(['squeue','-j',job,'--noheader','-o','%i|%T|%M|%R'],text=True),accounting=subprocess.check_output(['sacct','-j',job,'--parsable2','--noheader','--format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES'],text=True),predictions=sum(1 for _ in pred.open()) if pred.exists() else 0,completed=json.loads((out/'completed.json').read_text()) if (out/'completed.json').exists() else None,stderr=(run/('slurm-'+job+'.err')).read_text()[-3000:] if (run/('slurm-'+job+'.err')).exists() else '')))
'''
    s=command(code);(OUT/('observation_'+datetime.now().strftime('%H%M%S')+'.json')).write_text(s,encoding='utf-8');print(s)
def fetch():
    job=json.loads((OUT/'submission_result.json').read_text())['job_id'];code='job='+repr(job)+r'''
import json,base64,pathlib,hashlib
p=pathlib.Path.cwd();files=[]
for folder,prefix in [(p/('runs/2b-early9-'+job),'result'),(p/'runs/2b-early9-inputs-20261020-01','input')]:
 for src in folder.rglob('*'):
  if not src.is_file():continue
  value=src.read_bytes()
  if len(value)>8*2**20:raise ValueError('bounded evidence')
  files.append(dict(path=prefix+'/'+src.relative_to(folder).as_posix(),sha256=hashlib.sha256(value).hexdigest(),base64=base64.b64encode(value).decode()))
print(json.dumps(files))
'''
    result=json.loads(command(code));out=OUT/('gpu_'+job);out.mkdir(exist_ok=False)
    for item in result:
        value=base64.b64decode(item.pop('base64'));dst=out/item['path']
        if not dst.resolve().is_relative_to(out.resolve()) or hashlib.sha256(value).hexdigest()!=item['sha256']:raise ValueError('fetch integrity')
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(value)
    save(OUT/'fetch_receipt.json',result);print(json.dumps(dict(files=len(result),output=str(out))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','install_cpu','submit','observe','fetch']);a=p.parse_args();globals()[a.mode]()
