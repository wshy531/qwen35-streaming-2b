"""Read-only CPU media checks in the authorized project."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/stream_prune_cost_prepare_20261102_01'
sys.path.insert(0,str(ROOT/'experiments/streamingbench-router-v1'))
from v0_source import V0Pairs

questions = json.loads((PREP/'panel.json').read_text(encoding='utf-8'))
rows = []
started = time.monotonic()
for q in questions:
    path = (ROOT/q['media_path']).resolve()
    assert path.is_relative_to(ROOT.resolve())
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle,'sha256').hexdigest()
    assert digest == q['media_sha256']
    if 'media_bytes' in q:
        assert path.stat().st_size == q['media_bytes']
    rows.append(dict(path=q['media_path'], sha256=digest, bytes=path.stat().st_size))
q = min(questions,key=lambda x:x['query_time_s'])
pairs = []
driver = V0Pairs(ROOT/q['media_path'], lambda first,second,padding: pairs.append((first.frame_id,second.frame_id,first.timestamp,second.timestamp,padding)), config='auto4')
try:
    state = driver.advance(q['query_time_s'])
finally:
    driver.close()
refs = json.loads((PREP/'reference.json').read_text(encoding='utf-8'))
reference = next(r for r in refs if r['id']==q['id'])
assert state == reference['source_state']
assert all(p[3]<=q['query_time_s']+1e-6 for p in pairs)
result = dict(status='passed', rows=rows, short_source_state_match=True, short_pairs=len(pairs), seconds=time.monotonic()-started, GPU=0)
with (PREP/'media_verified.json').open('x',encoding='utf-8') as handle:
    json.dump(result,handle,indent=2)
print(json.dumps(dict(status='passed', files=len(rows), short_pairs=len(pairs), seconds=time.monotonic()-started)))
