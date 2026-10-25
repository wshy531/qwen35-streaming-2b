"""Auditable token/workspace projections from historical processor receipts."""
from .controls import geometry
from .native import CONTEXT_NATIVE,QUERY_TILE


def forecast(checks,selections):
    pairs={r['id']:len(r['B'])//2 for r in selections}
    if len(pairs)!=len(selections) or any(len(r['B'])%2 for r in selections):
        raise ValueError('unique paired evidence required')
    indexed={(r['id'],r['arm'],r['shift']):r['input_tokens'] for r in checks}
    if len(indexed)!=len(checks): raise ValueError('duplicate historical input')
    predicted=[]
    for row in checks:
        if row['arm']!='R': continue
        qid,s=row['id'],row['shift']; count=pairs[qid]
        low=indexed[qid,'B',s]
        if row['input_tokens']-low!=count*(geometry('mid')['pair_tokens']-geometry('low')['pair_tokens']):
            raise ValueError('historical token delta is not explained by the spatial grid')
        length=row['input_tokens']+count*(geometry('high')['pair_tokens']-geometry('mid')['pair_tokens'])
        if length+8>CONTEXT_NATIVE: raise ValueError('projected input exceeds approved context')
        predicted.append(dict(id=qid,shift=s,input_tokens=length,pairs=count))
    if not predicted: raise ValueError('missing native high forecasts')
    longest=max(predicted,key=lambda r:r['input_tokens']); n=longest['input_tokens']; heads=8
    return dict(basis='historical token delta checked for every question/rotation; CPU must confirm projection',
        new_high_inputs=predicted,maximum_input_tokens=n,maximum_with_generation=n+8,
        mean_input_tokens=sum(r['input_tokens'] for r in predicted)/len(predicted),
        longest_question=longest['id'],maximum_pairs=max(pairs.values()),
        questions_over_old_24K=len({r['id'] for r in predicted if r['input_tokens']+8>24000}),
        pairs_per_resolution=sum(pairs.values()),memory_visual_forwards=3*sum(pairs.values()),
        total_visual_forwards=690+3*sum(pairs.values()),
        workspace=dict(heads=heads,bytes_per_element=4,query_tile=QUERY_TILE,
            one_full_attention_matrix_GiB=heads*n*n*4/2**30,
            two_full_attention_matrices_GiB=2*heads*n*n*4/2**30,
            one_tiled_attention_matrix_GiB=heads*QUERY_TILE*n*4/2**30,
            one_dense_causal_mask_GiB=n*n*4/2**30,
            caveat='tensor size calculations, not a measured or guaranteed total peak; weights, cache, linear/vision workspace also exist'),
        bank_resident_bytes={r:geometry(r)['total_tokens']*(2048*4+28)+2048*4 for r in ('low','mid','high')},
        proposed=dict(GPUs=1,CPUs=4,RAM_GiB=48,minutes=360,allocator_cap_GiB=72,
            free_GPU_required_GiB=64,RSS_cap_GiB=40,models_resident=1,pretrained_loads=2),
        two_GPU_assessment='layer sharding does not divide a single attention workspace; no unvalidated tensor parallelism or automatic second allocation')
