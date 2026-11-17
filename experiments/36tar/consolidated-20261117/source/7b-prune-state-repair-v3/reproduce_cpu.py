import sys,json,hashlib,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-memory-transfer-v1'))
from video_source import CausalPairs
from state_contract import compare_state,NEW_CANVAS,FIELD

out=ROOT/'runs/7b_prune_state_repair_20261111_01'
trace=json.loads((ROOT/'runs/7b_method_transfer_prepare_20261110_01/traces.json').read_text(encoding='utf8'))
spec=trace['SB150|sample_102_real.mp4'];qid='streamingbench:90dfbb0c7d53033fcd002de6456d2b739f583db5ba94b06c6705f8ae10cf6014'
media=out/'sample_102_real.mp4'
assert hashlib.sha256(media.read_bytes()).hexdigest()=='fd0a08af04a3e9541eb32a591f51cfd4987a48c3a51b87b829dd635d64e71d7c'
pairs=[]
def commit(a,b,padding=False):
    assert a.rgb.shape==b.rgb.shape==NEW_CANVAS
    pairs.append(dict(first=a.frame_id,second=b.frame_id,first_pts=a.timestamp,second_pts=b.timestamp))
t=time.monotonic();driver=CausalPairs(media,commit)
try:actual=driver.advance(281.0)
finally:driver.close()
ref=spec['states'][qid]
assert actual!=ref # Reproduce the exact old assertion failure before the repair.
result=compare_state(actual,ref)
# A canvas translation must never mask time/sample/EOF errors or an unknown canvas.
rejected=[]
for k in actual:
    bad=copy.deepcopy(actual)
    bad[k]=(not bad[k]) if isinstance(bad[k],bool) else (bad[k]+1)
    try:compare_state(bad,ref)
    except AssertionError:rejected.append(k)
    else:raise AssertionError(('corrupted state accepted',k))
assert len(rejected)==len(actual)
result.update(id=qid,actual=actual,reference=ref,pairs=len(pairs),
    rejected_corrupted_fields=rejected,seconds=time.monotonic()-t,GPU=0,model_loaded=False)
(out/'CPU_real_failure_reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result))
