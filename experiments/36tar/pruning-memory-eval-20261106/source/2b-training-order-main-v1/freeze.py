import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/training_order_main_prepare_20261104_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(name):
    p=ROOT/name
    if not p.exists() and name.startswith('repo/'):
        p=ROOT/'publish/qwen35-streaming-2b'/name[5:]
    return p
def main():
    inherited=json.loads((ROOT/'runs/training_order_screen_20261104_01/execution_lock.json').read_text())
    for name,h in inherited.items():assert sha(local(name))==h,name
    names=set(inherited)
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/2b-training-order-main-v1').glob('*.py'))
    names.update(['experiments/2b-training-order-main-v1/run_main.sh'])
    names.update(p.relative_to(ROOT).as_posix() for p in PREP.glob('*.json') if p.name in ('panel_LOCAL_ONLY.json','protocol.json','exclusions.json'))
    lock={name:sha(local(name)) for name in sorted(names)}
    (PREP/'execution_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    # Existing dependencies are verified remotely; transmit only the new files.
    send=[n for n in lock if n.startswith('experiments/2b-training-order-main-v1/') or n.startswith('runs/training_order_main_prepare_20261104_01/')]
    send+=['runs/training_order_main_prepare_20261104_01/execution_lock.json']
    with tarfile.open(PREP/'source.tar','w') as tar:
        for name in send:tar.add(ROOT/name,arcname=name)
    print(json.dumps(dict(files=len(lock),transfer=len(send),sha256=sha(PREP/'source.tar'))))
if __name__=='__main__':main()
