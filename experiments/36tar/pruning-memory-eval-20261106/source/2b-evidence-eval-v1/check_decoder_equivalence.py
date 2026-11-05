"""Real-frame CPU equivalence at a fixed 7-second prefix, no model or labels."""
import sys,json,hashlib,os
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
sys.path.insert(0,str(R/'experiments/streamingbench-interface-v1'))
sys.path.insert(0,str(R/'experiments/streamingbench-router-v1'))
from source import CausalPairs
from v0_source import V0Pairs
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
q=min((q for q in json.loads((P/'panel_media_bound.json').read_text()) if q['dataset']=='StreamingBench'),key=lambda q:q['query_time_s'])
results=[]
for cls in (CausalPairs,V0Pairs):
    rows=[]
    def commit(a,b,padding):
        rows.append((a.frame_id,a.timestamp,hashlib.sha256(a.rgb.tobytes()).hexdigest(),
                     b.frame_id,b.timestamp,hashlib.sha256(b.rgb.tobytes()).hexdigest(),padding))
    driver=cls(R/q['media_path'],commit)
    try:state=driver.advance(q['query_time_s'])
    finally:driver.close()
    results.append((rows,state))
assert results[0]==results[1]
out=dict(status='exact_pair_pixels_and_state',source=q['source_id'],query=q['query_time_s'],pairs=len(results[0][0]),GPU=0,API=0)
(P/'CPU_decoder_equivalence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
