from experiment import *
lock=read(R/'runs/anchor_merge_development_20261103_01/execution_lock.json')
for n,h in lock.items():
    p=R/n
    if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
paths=[*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.sh'),R/'experiments/2b-protection-prototypes-v1/protected_cache.py']
paths += [R/n for n in read(P/'loaded_files.json')]
paths += [R/'experiments/2b-midprune-memory-v1'/n for n in ('pilot.py','adapter.py')]
paths += [P/n for n in ('protocol.json','requests.json','videos.json','expected_mid_prefixes.json','paired_questions.json','paired_sources.json','baseline.json','CPU_local.json')]
for p in paths:
    n=p.relative_to(R).as_posix();n='repo/'+n[len('publish/qwen35-streaming-2b/'):] if n.startswith('publish/qwen35-streaming-2b/') else n
    lock[n]=sha(p)
save(P/'execution_lock.json',lock);print(len(lock),sha(P/'execution_lock.json'))

