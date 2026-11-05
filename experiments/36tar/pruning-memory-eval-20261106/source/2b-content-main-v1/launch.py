"""One quality shard or a fixed sequence of independent cost processes; no retries."""
import os,sys,time,subprocess,argparse
from common import ROOT,PREP,read,save,verify
def main(kind,index):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('approved Slurm required')
    verify();tasks=read(PREP/'tasks.json');minutes=40 if kind=='quality' else 35
    deadline=time.time()+minutes*60-120
    indices=[index] if kind=='quality' else list(range(2,len(tasks)))
    out=ROOT/'runs'/f'content-main-{kind}-{os.environ["SLURM_JOB_ID"]}';out.mkdir(exist_ok=False)
    completed=[]
    for i in indices:
        task=tasks[i]
        subprocess.run([sys.executable,str(ROOT/'experiments/2b-content-main-v1/worker.py'),
            '--task',str(i),'--out',str(out/task['name']),'--deadline',str(deadline)],check=True,cwd=ROOT)
        completed.append(read(out/task['name']/'completed.json'))
    save(out/'completed.json',dict(status='complete',kind=kind,processes=len(completed),
        answers=sum(x['answers'] for x in completed),warmup_answers=sum(x['warmup_answers'] for x in completed),
        generated_tokens=sum(x['generated_tokens'] for x in completed),
        vision_calls=sum(x['vision_calls']+x['warmup_vision_calls'] for x in completed),
        pretrained_loads=len(completed),automatic_retry=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['quality','cost'],required=True);p.add_argument('--index',type=int,default=0)
    a=p.parse_args();main(a.kind,a.index)
