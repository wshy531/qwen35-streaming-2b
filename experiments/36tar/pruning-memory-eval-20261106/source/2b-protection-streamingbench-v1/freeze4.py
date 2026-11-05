from fair_common import *
assert read(P/'CPU4_local.json')==read(P/'CPU4_cluster.json')
lock=read(P/'execution_lock.json')
for n,h in lock.items():
    p=R/n
    if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+[P/'CPU4_local.json']
for p in files:lock[p.relative_to(R).as_posix()]=sha(p)
save(P/'execution_lock4.json',lock);print('LOCK4',len(lock),sha(P/'execution_lock4.json'))
