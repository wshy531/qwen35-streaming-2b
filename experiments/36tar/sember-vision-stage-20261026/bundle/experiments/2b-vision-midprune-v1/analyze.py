"""Apply the predeclared visual-cost gate; no question accuracy inferred."""
import argparse
from pathlib import Path
import statistics
from common import OUT,read,save,verify


def main(results):
    protocol=verify();result=read(results/'completed.json')
    assert result['status']=='complete' and result['vision_forwards']==924 and len(result['rows'])==12
    rows=result['rows'];ratios=[r['mid_vs_original_ratio'] for r in rows]
    for r in rows:
        assert r['all_kept_max_abs_error']<=protocol['gate']['all_kept_gpu_max_abs_error']
        assert all(len(v)==20 for v in r['samples_seconds'].values())
        assert r['output_shapes']==dict(original_full=[108,2048],post_odd=[54,2048],mid_odd=[54,2048])
    median=statistics.median(ratios);slower=sum(r>1 for r in ratios)
    gate=median<=.85 and slower<=3
    save(OUT/'visual_cost_analysis.json',dict(status='complete',median_pair_ratio=median,pairs_slower_than_original=slower,
        per_pair_ratios=ratios,gate_passed=gate,activate_R2_quality_preparation=gate,
        question_accuracy_not_measured=True,end_to_end_not_measured=True,default_changed=False))
    print(dict(status='complete',median_pair_ratio=median,pairs_slower_than_original=slower,gate_passed=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    main(p.parse_args().results)
