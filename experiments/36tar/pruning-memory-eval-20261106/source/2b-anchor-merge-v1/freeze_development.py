from development_worker import R,P,old,read,save
from pathlib import Path
lock=read(R/'runs/prune_timing_prepare_20261103_01/execution_lock.json')
for name,h in lock.items():
    p=R/name
    if name.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/name[5:]
    assert old.sha(p)==h,name
for name in read(P/'CPU_local.json')['loaded_files']:
    canonical='repo/'+name[len('publish/qwen35-streaming-2b/'):] if name.startswith('publish/qwen35-streaming-2b/') else name
    lock[canonical]=old.sha(R/name)
paths=[*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.sh'),R/'experiments/2b-protection-prototypes-v1/anchor_merge.py']
paths += [P/n for n in ['questions.json','sources.json','samples.json','baseline.json','protocol.json']]
for p in paths:lock[p.relative_to(R).as_posix()]=old.sha(p)
save(P/'execution_lock.json',lock)
print(len(lock),old.sha(P/'execution_lock.json'))
