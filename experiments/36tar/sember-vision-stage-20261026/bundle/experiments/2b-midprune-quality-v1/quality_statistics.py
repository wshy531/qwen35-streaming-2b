"""Source-paired exploratory bootstrap, same procedure/seed as prior formal analysis."""
import numpy as np


def grouped_interval(qids,requests,difference):
    sources=sorted({q['parent_source'] for q in requests.values()});index={s:i for i,s in enumerate(sources)}
    counts=np.zeros(len(sources));deltas=np.zeros(len(sources))
    for qid in qids:
        i=index[requests[qid]['parent_source']];counts[i]+=1;deltas[i]+=difference[qid]
    rng=np.random.default_rng(20261025);sample=rng.integers(0,len(sources),size=(5000,len(sources)))
    denominator=counts[sample].sum(1);valid=denominator>0;values=deltas[sample].sum(1)[valid]/denominator[valid]
    if not len(values):return dict(lower=None,upper=None,valid_resamples=0)
    lower,upper=np.quantile(values,[.025,.975])
    return dict(lower=float(lower),upper=float(upper),valid_resamples=int(valid.sum()),
        grouping='original parent video source; participant independence unknown',seed=20261025,resamples=5000,
        paired=True,multiple_comparison_adjustment=False,not_statistical_noninferiority_proof=True)
