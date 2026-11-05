"""No weights/GPU: full hashes, original causal sampling, all submitted pixel digests."""
import os,time,collections
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.name!='nt'
groups=collections.defaultdict(list)
for q in read(P/'questions.json'):groups[q['source_id']].append(q)
start=time.monotonic();prefixes={};total=0
for index,(source,qs) in enumerate(sorted(groups.items())):
    q=qs[0];path=R/q['media_path'];assert path.resolve().is_relative_to((R/'datasets/StreamingBench-eval200-v1').resolve())
    assert path.stat().st_size==q['media_bytes'] and sha(path)==q['media_sha256'],source
    d=hashlib.sha256();pairs=[0]
    def commit(a,b,padding):
        pair_digest(d,a,b,padding);pairs[0]+=1
    driver=V0Pairs(path,commit,config='auto4')
    try:
        for q in sorted(qs,key=lambda q:(q['query_time_s'],q['id'])):
            state=driver.advance(q['query_time_s'])
            assert not state['future_RGB_emitted'] and state['complete_pairs']>0
            assert state['last_committed_pts']<=q['query_time_s']+1e-6
            prefixes[q['id']]=dict(state=state,pixel_sha256=d.hexdigest())
    finally:driver.close()
    total+=pairs[0]
    if (index+1)%12==0:print(json.dumps(dict(sources=index+1,pairs=total,seconds=time.monotonic()-start)),flush=True)
save(P/'media_verified.json',dict(passed=True,sources=len(groups),questions=len(prefixes),vision_pairs=total,prefixes=prefixes,seconds=time.monotonic()-start))
print('MEDIA_COMPLETE',total,flush=True)
