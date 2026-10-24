"""Nine sequential independent workers, no submission and no retry."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,read,save,sha


def main(run,out):
    auth=read(run/'GPU_authorization.json');lock=read(run/'execution_lock.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(run/'execution_lock.json')
    assert os.environ.get('SLURM_JOB_ID') and out.resolve().is_relative_to((ROOT/'runs').resolve())
    out.mkdir(exist_ok=False);started=time.monotonic();jobs=read(run/'jobs.json');results=[]
    for index,job in enumerate(jobs):
        remaining=3300-(time.monotonic()-started);assert remaining>30
        target=out/('block'+str(job['block'])+'_'+job['arm'])
        command=[sys.executable,str(Path(__file__).parent/'worker.py'),'--run',str(run),'--out',str(target),'--job-index',str(index)]
        with (out/('worker-'+str(index)+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
        assert result.returncode==0,dict(worker=index,exit_code=result.returncode)
        done=read(target/'completed.json');startup=read(target/'startup.json')
        results.append(dict(directory=target.name,completed=done,startup=startup))
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<64*2**20
        print(dict(workers_completed=len(results),measured_answers=sum(r['completed']['measured_answers'] for r in results)),flush=True)
    for name in ('answers.jsonl','tokens.jsonl'):
        with (out/name).open('x',encoding='utf-8') as destination:
            for r in results:
                with (out/r['directory']/name).open(encoding='utf-8') as source:
                    for line in source:destination.write(line)
    assert len(results)==6
    assert sum(r['completed']['measured_answers'] for r in results)==228
    assert sum(r['completed']['measured_encodings'] for r in results)==read(run/'protocol.json')['measured_visual_encodings']
    assert sum(r['startup']['warmup_encodings'] for r in results)==48
    generated=sum(r['completed']['measured_generated_tokens']+r['startup']['warmup_generated_tokens'] for r in results)
    assert generated<=1872 and sum(r['startup']['pretrained_loads'] for r in results)==6
    save(out/'completed.json',dict(status='complete',workers=6,measured_answers=228,warmup_answers=6,
         generated_tokens=generated,measured_encodings=read(run/'protocol.json')['measured_visual_encodings'],warmup_encodings=48,pretrained_loads=6,
         wall_seconds=time.monotonic()-started,results=results,execution_lock_sha256=sha(run/'execution_lock.json'),
         actual_device='A800 cluster',not_edge_hardware=True,not_wallclock_1x=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.run,a.out)
