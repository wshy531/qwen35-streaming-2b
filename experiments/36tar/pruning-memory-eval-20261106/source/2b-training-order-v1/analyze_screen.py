"""Score fixed checkpoints and paired evidence controls without ranking by CE."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['id']].append(row)
    assert all({r['shift'] for r in group}=={0,2} and len(group)==2 for group in groups.values())
    return dict(correct=sum(r['correct'] for r in rows), answers=len(rows),
                both_orders_correct=sum(all(r['correct'] for r in group) for group in groups.values()),
                questions=len(groups), sources=len({r['video_id'] for r in rows}),
                mean_ce=mean(r['ce'] for r in rows) if rows else None)


def paired(before, after):
    key = lambda r: (r['id'],r['shift'])
    left, right = {key(r):r for r in before}, {key(r):r for r in after}
    assert len(left)==len(before) and len(right)==len(after) and left.keys()==right.keys()
    return dict(gains=[list(k) for k in left if not left[k]['correct'] and right[k]['correct']],
                losses=[list(k) for k in left if left[k]['correct'] and not right[k]['correct']])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    args = p.parse_args()
    run = json.loads((args.run/'completed.json').read_text())
    assert run['status']=='completed' and not run['cpu_tiny']
    updates = [json.loads(line) for line in (args.run/'updates.jsonl').read_text().splitlines()]
    evidence = defaultdict(list)
    for update in updates:
        evidence[update['arm']].extend(s for s in update['samples'] if s['kind']=='delayed')
    metrics = {}
    end = {}
    for name, evaluation in run['evaluations'].items():
        rows = evaluation['answers']
        by_type = {t:[r for r in rows if r['readout']==t] for t in ('short','delayed','late_recent')}
        metrics[name] = {k:summarize(v) for k,v in by_type.items() if v}
        if name=='initial' or name.endswith('step8'):
            end[name] = by_type
    comparisons = {}
    for name, by_type in end.items():
        comparisons[name] = dict(
            delayed_vs_late_recent=paired(by_type['late_recent'],by_type['delayed']),
            delayed_vs_text=paired(run['text'],by_type['delayed']),
            fresh={k:summarize([r for r in v if not r['prior_preflight']]) for k,v in by_type.items()},
            changes_from_initial={k:paired(end['initial'][k],v) for k,v in by_type.items()},
        )
    comparison = {k:paired(end['mixed_step8'][k],v) for k,v in end['short_then_delayed_step8'].items()}
    result = dict(text=summarize(run['text']), checkpoints=metrics, endpoints=comparisons,
                  sequence_vs_mixed_final=comparison,
                  peak_allocated_gib=max(r['peak_allocated_bytes'] for r in updates)/2**30,
                  delayed_train={arm:dict(exposures=len(rows),
                      merged_exposures=sum(bool(r['merged_evidence_pairs']) for r in rows),
                      nonzero_merged_gradient=sum(r['merged_evidence_grad_l1']>0 for r in rows))
                      for arm,rows in evidence.items()})
    with (args.run/'analysis.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(text=result['text'], final={k:metrics[k] for k in end},
                          peak_allocated_gib=result['peak_allocated_gib'],
                          delayed_train=result['delayed_train']), indent=2))


if __name__ == '__main__':
    main()
