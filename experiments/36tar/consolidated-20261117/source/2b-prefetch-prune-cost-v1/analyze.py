"""Source-paired cost analysis; repeats are not extra independent video sources."""
import argparse
import json
from pathlib import Path
import random
import statistics
from common import CASES, PREP, read, write

def summarize(rows, panel):
    expected={(q['id'],case,repeat) for q in panel for case in CASES for repeat in range(4)}
    keys=[(r['id'],r['case'],r['repeat']) for r in rows]
    assert len(keys)==len(set(keys))==96 and set(keys)==expected
    assert all(all(r['reference_match'].values()) for r in rows)
    medians={case:{q['id']:statistics.median(r['pipeline_wall_seconds'] for r in rows if r['id']==q['id'] and r['case']==case)
                  for q in panel} for case in CASES}
    totals={case:sum(vals.values()) for case,vals in medians.items()}
    comparisons=[]
    for base, candidate in [('serial_full','prefetch_full'),('prefetch_full','prefetch_fixed'),
                             ('prefetch_full','prefetch_content'),('prefetch_fixed','prefetch_content')]:
        ids=[q['id'] for q in panel];rng=random.Random(1109);samples=[]
        for _ in range(5000):
            chosen=rng.choices(ids,k=len(ids))
            samples.append(1-sum(medians[candidate][k] for k in chosen)/sum(medians[base][k] for k in chosen))
        samples.sort();saving=1-totals[candidate]/totals[base]
        faster=sum(medians[candidate][k]<medians[base][k] for k in ids)
        comparisons.append(dict(baseline=base,candidate=candidate,reduction_fraction=saving,
              faster_sources=faster,source_bootstrap_exploratory_95=[samples[125],samples[4874]],
              meets_practical_cost_screen=(saving>=.05 and faster>=5) if base=='prefetch_full' else None))
    return dict(status='complete_cost_panel',sources=6,repeats=4,formal_answers=96,
          median_pipeline_by_source=medians,sum_source_median_pipeline_seconds=totals,comparisons=comparisons,
          limitation='No new capability or edge claim. Fixed six opened sources; no cold-cache control; selection overhead included.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
    assert read(a.out/'completed.json')['answers']==96
    rows=[json.loads(s) for s in (a.out/'answers.jsonl').read_text(encoding='utf-8').splitlines() if s]
    write(a.out/'analysis.json',summarize(rows,read(PREP/'panel.json')))
