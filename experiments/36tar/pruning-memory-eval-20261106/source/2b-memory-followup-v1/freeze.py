from followup import *
lock=read(P/'execution_lock4.json')
for n,h in lock.items():
    p=R/n
    if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))
files += [OUT/'manifest.json',OUT/'CPU_local.json',HEAD_PATH,R/'scripts/router_head_predict.py',
    R/'experiments/2b-read-benefit-v1/feature_runtime.py',R/'experiments/2b-sember-v1/sember_runtime.py',
    R/'experiments/2b-sember-v1/sember_nodes.py']
for p in files:lock[p.relative_to(R).as_posix()]=sha(p)
save(OUT/'execution_lock.json',lock);print(len(lock),sha(OUT/'execution_lock.json'))
