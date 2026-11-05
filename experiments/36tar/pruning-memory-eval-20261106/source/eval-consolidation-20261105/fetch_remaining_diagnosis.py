"""Relay the approved diagnostic ZIP members and verify each cluster copy."""
import argparse
import hashlib
import json
import re
import subprocess
import time

import httpx

from fetch_validation import ROOT, get_stream, read, save

REMOTE = '/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
RUN = ROOT / 'runs/diagnosis_expansion_download_20261105_01'


def ssh(code, timeout=120):
    p = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'ChaoSuan','python -'],input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError('Remote integrity or path check failed')
    return p.stdout.strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    path=ROOT/'runs/eval_consolidation_20261105_03/diagnosis_expansion_plan.json'
    plan=read(path);files=plan['files']
    expected=sum(q['member']['compressed_bytes']+1024 for q in files)
    assert len(files)==len({q['name'] for q in files})==26
    assert expected==plan['bytes']==7137207045 and plan['cap']==7*2**30
    assert plan['maximum_seconds']==3600 and plan['threads']==1 and plan['retries']==0
    assert all(re.fullmatch(r'sample_\d+_real\.mp4',q['name']) for q in files)
    if not args.execute:
        print(json.dumps(dict(videos=26,bytes=expected,GPU=0,cap=plan['cap'],
                              seconds=3600,mode='local_relay_dry_run_no_network')))
        return
    target=ROOT/plan['target']
    assert target.resolve().is_relative_to(ROOT.resolve())
    assert all(not (target/q['name']).exists() and not (target/q['name']).with_suffix('.partial').exists() for q in files)
    RUN.mkdir(exist_ok=False)
    target.mkdir(exist_ok=True)
    names=[q['name'] for q in files]
    check=f"""from pathlib import Path
r=Path({REMOTE!r});d=r/{plan['target']!r}
assert d.resolve().is_relative_to(r.resolve()) and d.is_dir()
assert all(not (d/n).exists() and not (d/(n+'.uploading')).exists() for n in {names!r})
print('ready')
"""
    assert ssh(check)=='ready'
    start=time.monotonic()
    state=dict(status='running',wire_bytes=0,complete=[],expected_bytes=expected,
               plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def checkpoint():
        state['seconds']=time.monotonic()-start
        save(RUN/'download.json',state)
    def guard():
        if state['wire_bytes']>plan['cap'] or time.monotonic()-start>plan['maximum_seconds']:
            raise TimeoutError('Approved budget exhausted')
    def chunks(response):
        for data in response.iter_bytes(262144):
            state['wire_bytes']+=len(data);guard();yield data
    base=f"https://huggingface.co/datasets/{plan['repo']}/resolve/{plan['revision']}/"
    checkpoint()
    try:
        with httpx.Client(timeout=httpx.Timeout(35,connect=10),follow_redirects=True) as client:
            for row in files:
                guard()
                destination=target/row['name']
                sha=get_stream(client,row,base,destination,chunks)
                assert destination.stat().st_size==row['member']['bytes']
                remote_part=f"{REMOTE}/{plan['target']}/{row['name']}.uploading"
                guard()
                remaining=max(1,int(plan['maximum_seconds']-(time.monotonic()-start)))
                subprocess.run(['scp','-o','BatchMode=yes',str(destination),'ChaoSuan:'+remote_part],
                               check=True,timeout=min(300,remaining))
                code=f"""from pathlib import Path
import hashlib
r=Path({REMOTE!r});d=r/{plan['target']!r};p=d/({row['name']!r}+'.uploading');final=d/{row['name']!r}
assert p.resolve().is_relative_to(r.resolve()) and final.resolve().is_relative_to(r.resolve())
assert not p.is_symlink() and not final.exists() and p.stat().st_size=={row['member']['bytes']}
h=hashlib.sha256()
with p.open('rb') as f:
 for data in iter(lambda:f.read(2**20),b''):h.update(data)
assert h.hexdigest()=={sha!r}
p.rename(final)
print('verified')
"""
                remaining=max(1,int(plan['maximum_seconds']-(time.monotonic()-start)))
                assert ssh(code,timeout=min(120,remaining))=='verified'
                guard()
                state['complete'].append(dict(name=row['name'],bytes=destination.stat().st_size,
                    sha256=sha,media_path=destination.relative_to(ROOT).as_posix(),cluster_sha_verified=True))
                checkpoint()
                print(f"{len(state['complete'])}/26 verified on cluster",flush=True)
        state['status']='complete'
    except Exception as error:
        state.update(status='stopped',error_type=type(error).__name__)
    finally:
        checkpoint()
    if state['status']!='complete':
        raise SystemExit(2)


if __name__=='__main__':
    main()
