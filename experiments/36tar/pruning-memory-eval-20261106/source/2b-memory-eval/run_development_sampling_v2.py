"""Bound the one planned development sampling audit, no GPU or model import."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=Path(sys.argv[1]).resolve()
    if out.parent!=(ROOT/'runs').resolve():raise ValueError('own output required')
    logpath=out.with_suffix('.log');start=time.monotonic();peak=0;stop=None
    if out.exists() or logpath.exists():raise ValueError('no output overwrite')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1')
    with logpath.open('x',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('audit_development_sampling_v2.py')),
            '--out',str(out)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        owner=psutil.Process(proc.pid);members=[]
        while proc.poll() is None:
            try:
                members=[owner,*owner.children(recursive=True)]
                peak=max(peak,sum(m.memory_info().rss for m in members if m.is_running()))
            except psutil.NoSuchProcess:pass
            if peak>512*2**20 or time.monotonic()-start>600 or psutil.virtual_memory().available<.5*2**30:
                stop='own_CPU_RSS_wall_or_system_reserve'
                for m in reversed(members):
                    try:m.kill()
                    except psutil.NoSuchProcess:pass
                break
            time.sleep(.2)
        code=proc.wait()
    receipt=dict(exit_code=code,stop_reason=stop,elapsed_seconds=time.monotonic()-start,
        peak_sampled_owned_tree_rss_bytes=peak,CUDA_VISIBLE_DEVICES='-1',model_loads=0,GPU_runs=0)
    out.mkdir(exist_ok=True)
    (out/'supervisor_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
