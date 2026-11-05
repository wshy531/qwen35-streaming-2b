import os,time
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
seen={};rows=[];start=time.monotonic()
for q in read(P/'panel.json'):
    name=q['media_path']
    if name in seen:assert seen[name]==q['media_sha256'];continue
    path=R/name;assert path.resolve().is_relative_to(R.resolve());assert sha(path)==q['media_sha256'],name
    seen[name]=q['media_sha256'];rows.append(dict(path=name,sha256=q['media_sha256'],bytes=path.stat().st_size))
save(P/'media_verified_REMOTE.json',dict(files=len(rows),rows=rows,seconds=time.monotonic()-start,GPU=0))
print(json.dumps(dict(files=len(rows),seconds=time.monotonic()-start)))
