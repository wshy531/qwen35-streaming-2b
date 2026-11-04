"""Causal recent-only feature schedule; no options, answer labels or history."""
from collections import defaultdict

def schedule(requests, resolution, sources):
    expected={'id','question','recent_frames','query_time_s'}
    if len({r['id'] for r in requests})!=len(requests):raise ValueError('duplicate request')
    metadata={s['name']:s for s in sources}
    if len(metadata)!=len(sources) or set(resolution)!={r['id'] for r in requests}:raise ValueError('source mapping mismatch')
    by_source=defaultdict(list)
    for r in requests:
        if set(r)!=expected:raise ValueError('non-allowlisted feature request')
        binding=resolution[r['id']];source=metadata[binding['source_name']]
        if binding['sha256']!=source['sha256']:raise ValueError('media identity drift')
        frames=r['recent_frames']
        if len(frames)!=8 or any(f['pts_s']>r['query_time_s'] for f in frames):raise ValueError('noncausal or incomplete frames')
        lookup={tuple(f['frame_id'] for f in p['frames']):p for p in source['pairs']}
        selected=[]
        for i in range(4):
            ff=frames[2*i:2*i+2];pair=lookup.get(tuple(f['frame_id'] for f in ff))
            if pair is None or pair['frames']!=ff:raise ValueError('fixed pair pixels differ')
            selected.append(pair['index'])
        if selected!=list(range(selected[-1]-3,selected[-1]+1)):raise ValueError('recent sequence not contiguous')
        by_source[source['name']].append(dict(request=r,last_pair=selected[-1],pair_indices=selected))
    plans=[]
    for source in sources:
        jobs=by_source[source['name']]
        if not jobs:raise ValueError('unused source')
        wanted={i for q in jobs for i in q['pair_indices']}
        if wanted!={p['index'] for p in source['pairs']}:raise ValueError('hidden unrequested history pairs')
        if [p['index'] for p in source['pairs']]!=sorted(wanted):raise ValueError('pair order changed')
        plans.append(dict(source=source,queries=jobs))
    return plans
