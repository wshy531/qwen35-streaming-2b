"""Reuse frozen V0 source checks, full media SHA; two fresh pixel walkthroughs."""
import os,time,collections
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.name!='nt'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
groups=collections.defaultdict(list);refs=read(P/'causal_reference.json');start=time.monotonic()
for q in read(P/'questions.json'):groups[q['source_id']].append(q)
files=[];prefixes=refs['prefixes'];checked=0
for index,(source,qs) in enumerate(sorted(groups.items())):
    q=qs[0];path=R/q['media_path'];assert path.resolve().is_relative_to((R/'datasets/StreamingBench-eval200-v1').resolve())
    if 'media_bytes' in q:assert path.stat().st_size==q['media_bytes']
    assert sha(path)==q['media_sha256'],source
    files.append(dict(path=q['media_path'],bytes=path.stat().st_size,sha256=q['media_sha256']))
    selected=[x for x in qs if x['id'] in refs['pixel_check_ids']]
    if selected:
        d=hashlib.sha256()
        def commit(a,b,padding):pair_digest(d,a,b,padding)
        driver=V0Pairs(path,commit,config='auto4')
        try:
            for q in sorted(selected,key=lambda q:(q['query_time_s'],q['id'])):
                state=driver.advance(q['query_time_s']);assert state==prefixes[q['id']]['state'],q['id']
                prefixes[q['id']]['pixel_sha256']=d.hexdigest();checked+=1
        finally:driver.close()
    if (index+1)%24==0:print('media hashes',index+1,flush=True)
assert checked==2
save(P/'media_verified.json',dict(passed=True,sources=len(groups),questions=len(prefixes),vision_pairs=refs['vision_pairs'],
    prefixes=prefixes,files=files,fresh_pixel_queries=checked,source_states_reused=150,seconds=time.monotonic()-start))
print('MEDIA_COMPLETE',refs['vision_pairs'],flush=True)
