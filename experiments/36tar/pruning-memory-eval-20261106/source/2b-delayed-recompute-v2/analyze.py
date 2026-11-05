"""Summarize completed fixed-pair numerical/cost checks without training claims."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    folder=args.folder
    completed=json.loads((folder/'completed.json').read_text())
    rows=[json.loads(line) for line in (folder/'cases.jsonl').read_text().splitlines()]
    assert len(rows)==2 and len({r['id'] for r in rows})==2
    assert completed['updates']==0 and completed['generated_tokens']==0
    summaries=[]
    for row in rows:
        arms=row['arms']; reference=arms['whole_repeat']; candidate=arms['recompute']
        assert reference['peak_allocated_bytes'] is not None
        summaries.append(dict(id=row['id'],passed=row['passed'],pairs=reference['pairs'],
            ce={k:v['ce'] for k,v in arms.items()},
            allocated_GiB={k:v['peak_allocated_bytes']/2**30 for k,v in arms.items()},
            reserved_GiB={k:v['peak_reserved_bytes']/2**30 for k,v in arms.items()},
            seconds={k:v['seconds'] for k,v in arms.items()},
            allocated_reduction_pct=100*(1-candidate['peak_allocated_bytes']/reference['peak_allocated_bytes']),
            compute_time_ratio=candidate['seconds']/reference['seconds'],
            max_gradient_abs={k:max(v['max_abs'] for v in vals.values()) for k,vals in row['comparisons'].items()},
            max_gradient_relative_l2={k:max(v['relative_l2'] for v in vals.values()) for k,vals in row['comparisons'].items()},
            failed_gradient_elements={k:sum(v['mismatched_elements'] for v in vals.values()) for k,vals in row['comparisons'].items()},
            cpu_cache_tensor_MiB=candidate['cache_tensor_bytes']/2**20,
            cpu_leaf_gradients_MiB=candidate['leaf_gradient_bytes']/2**20,
            qa_projected_leaves_MiB=candidate['projected_leaf_bytes']/2**20))
    report=dict(status=completed['status'],cases=summaries,
        interpretation='Fixed-pair engineering preflight. Timings exclude model load and decode; no warm-order balanced throughput study.',
        limitations=['Two previously used training examples, no parameter updates or generated QA.',
            'Fixed LS event schedule and existing STE; no derivative through discrete routing.',
            'CPU pixels, descriptors and projected features still grow with bounded clip length.',
            'Cannot infer broad training quality or long-video memory bounds from these two cases.'],
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [folder/'completed.json',folder/'cases.jsonl']})
    with (folder/'analysis.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
