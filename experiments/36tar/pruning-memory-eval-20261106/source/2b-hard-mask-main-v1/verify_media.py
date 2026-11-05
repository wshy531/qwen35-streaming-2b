"""CPU lock and complete media hashes; no model load or GPU."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/hard_mask_main_prepare_20261102_01'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
lock=read(P/'execution_lock.json')
for n,h in lock.items():
    path=R/n
    if not path.exists() and n.startswith('repo/'):path=R/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(path)==h,n
sources=read(R/'runs/learned_prune_main_prepare_20261101_01/sources.json')
for source in sources:
    assert sha(R/'datasets/learned-prune-nextqa-v1'/source['mirror_path'])==source['mirror_lfs_sha256'],source['video_id']
out=dict(execution_files=len(lock),media_files=len(sources),GPU=0)
(P/'verified_cluster.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
