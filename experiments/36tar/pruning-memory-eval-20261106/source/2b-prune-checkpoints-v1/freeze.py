"""Refresh only this experiment's pre-submission lock; refuse changes to inherited files."""
from prepare import R,P,read,sha,resolve,save
from pathlib import Path
old=read(R/'runs/hard_mask_main_prepare_20261102_01/execution_lock.json')
for n,h in old.items():assert sha(resolve(n))==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+list(Path(__file__).parent.glob('*.md'))
files += [P/n for n in ('questions.json','sources.json','reference.json','protocol.json')]
files += [R/n for n in read(P/'protocol.json')['checkpoints'].values()]
for p in files:old[p.relative_to(R).as_posix()]=sha(p)
save(P/'execution_lock.json',old)
print(dict(locked_files=len(old),sha256=sha(P/'execution_lock.json')))
