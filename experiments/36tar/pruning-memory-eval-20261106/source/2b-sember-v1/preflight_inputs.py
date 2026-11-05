"""Real PTS/RGB preflight only; no model, labels, or dependency requalification."""
from collections import defaultdict
import argparse
import time
from sember_common import ROOT,OUT,PREP,DATA,read,save
from sember_source import CausalSource,PairDriver


def check_prefix_identity(videos):
    # Eight fixed category profiler requests; repeated sources checked once.
    ids=sorted({q['video_id'] for q in read(PREP/'profiler_requests.json')})
    fingerprints={}
    for vid in ids:
        results=[]
        for config in ('auto4','slice1'):
            source=CausalSource(DATA/('videos/'+vid+'.mp4'),config=config)
            try:results.append([p.record() for p in source.advance(12.)])
            finally:source.close()
        assert results[0]==results[1],vid;fingerprints[vid]=results[0]
    return fingerprints


def main(scope):
    questions=read(PREP/'selection/requests.json');videos=read(PREP/'selection/videos.json')
    if scope=='development':questions=[q for q in questions if q['split']=='development18']
    grouped=defaultdict(list)
    for q in questions:grouped[q['video_id']].append(q)
    result=[];started=time.monotonic();bounds=0
    identity=check_prefix_identity(videos)
    for vi,vid in enumerate(sorted(grouped)):
        began=time.monotonic();source=CausalSource(DATA/('videos/'+vid+'.mp4'))
        pair_hashes=[]
        def append(a,b,*,padding):
            assert padding or (a.frame_id<b.frame_id and a.timestamp<b.timestamp)
            pair_hashes.append(dict(first=a.record(),second=b.record(),padding=padding))
            if len(pair_hashes)>512:raise RuntimeError('frozen pair metadata bound exceeded')
        driver=PairDriver(source,append);queries=[]
        try:
            for q in sorted(grouped[vid],key=lambda q:(q['question_time'],q['question_id'])):
                state=driver.advance(q['question_time'])
                assert state['pairs'] and state['last_committed_pts']<=q['question_time']+1e-9
                queries.append(dict(question_id=q['question_id'],**state))
                if time.monotonic()-began>180:raise TimeoutError('per-source CPU preflight limit reached')
            result.append(dict(video_id=vid,split=grouped[vid][0]['split'],queries=queries,source=source.status(),
                pair_records=pair_hashes,seconds=time.monotonic()-began))
        finally:driver.close()
        bounds+=len(pair_hashes)
        print(dict(sources_done=vi+1,total_sources=len(grouped),pairs_so_far=bounds,seconds=round(time.monotonic()-started,2)),flush=True)
    payload=dict(status='passed',scope=scope,sources=len(result),queries=len(questions),pairs=bounds,
        RGB_identity_prefix_sources=len(identity),RGB_identity_prefixes=identity,records=result,
        real_PTS_preflight=True,all_pairs_causal=True,full_frame_archive=False,no_model_outputs=True,
        no_dependency_requalification=True,GPU_jobs=0,seconds=time.monotonic()-started)
    OUT.mkdir(exist_ok=True);save(OUT/f'input_preflight_{scope}.json',payload)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['development','all'],default='development')
    main(parser.parse_args().scope)
