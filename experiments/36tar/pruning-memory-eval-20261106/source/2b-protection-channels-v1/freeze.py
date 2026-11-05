import json,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'experiments/2b-isolated-recent-v1'))
from experiment import read,save,sha
P=R/'runs/protection_channels_prepare_20261103_01';OLD=R/'runs/isolated_recent_prepare_20261103_01'
assert read(P/'CPU_local.json')==read(P/'CPU_cluster.json')
rows=[json.loads(x) for x in (OLD/'paired_snapshot_220254.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows)==270 and all(r['panel']=='paired' for r in rows)
baseline=[r for r in rows if r['arm'] in ('joint','isolated')];assert len(baseline)==180
save(P/'baseline.json',baseline)
save(P/'baseline_provenance.json',dict(source_job=220254,snapshot_sha256=sha(OLD/'paired_snapshot_220254.jsonl'),status='paired portion complete; full source job completion required before submission'))
lock=read(OLD/'execution_lock.json')
for n,h in lock.items():
    p=R/n
    if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))
files += [R/'experiments/2b-protection-prototypes-v1/channel_intervention.py',P/'baseline.json',P/'baseline_provenance.json',P/'CPU_local.json']
for p in files:lock[p.relative_to(R).as_posix()]=sha(p)
save(P/'execution_lock.json',lock);print('LOCK',len(lock),sha(P/'execution_lock.json'))
