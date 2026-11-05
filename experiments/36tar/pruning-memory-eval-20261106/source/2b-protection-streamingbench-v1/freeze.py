from common import *
assert read(P/'CPU_local.json')==read(P/'CPU_cluster.json')
assert read(P/'media_verified.json')['sources']==144
lock=read(R/'runs/isolated_recent_prepare_20261103_01/execution_lock.json')
for n,h in lock.items():
    p=R/n
    if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))
files += [R/'experiments/streamingbench-router-v1/v0_source.py',R/'experiments/streamingbench-interface-v1/source.py']
files += [P/n for n in ('questions.json','inventory.json','causal_reference.json','media_verified.json','CPU_local.json')]
for p in files:lock[p.relative_to(R).as_posix()]=sha(p)
save(P/'execution_lock.json',lock);print('LOCK',len(lock),sha(P/'execution_lock.json'))
