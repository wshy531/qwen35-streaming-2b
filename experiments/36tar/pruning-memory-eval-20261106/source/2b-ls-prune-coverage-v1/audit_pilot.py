"""Compare event boundaries and actual LS roots in the existing real pilot."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'runs/midprune-memory-220237'
OUT = ROOT / 'runs/ls_prune_coverage_prepare_20261103_01'
VIEWS = ('fixed12', 'heuristic11', 'mask_ce')


def rows(name):
    return [json.loads(line) for line in (OLD / name).read_text(encoding='utf-8').splitlines()]


def indexes(bank):
    nodes = bank['nodes']
    return ([(n['first'], n['last'], n['depth']) for n in nodes],
            {p for n in nodes for p in n['surviving_source_pair_ids']})


def main():
    trace = rows('event_trace.jsonl')
    events = {(r['source_name'], r['pair_index'], r['view']): r['starts_event'] for r in trace}
    answers = [r for r in rows('answers.jsonl') if r['policy'] == 'LS']
    by_question = {(r['id'], r['view']): r for r in answers}
    assert len(events) == len(trace) == 1716 and len(by_question) == len(answers) == 72
    full_events = {(source, i): v for (source, i, view), v in events.items() if view == 'full'}
    assert len(full_events) == 429
    counts = {}
    diagnostics = []
    for view in VIEWS:
        stats = Counter()
        for (source, i), full in full_events.items():
            pruned = events[source, i, view]
            stats['different_boundary'] += full != pruned
            stats['full_only_boundary'] += full and not pruned
            stats['pruned_only_boundary'] += pruned and not full
        for (qid, baseline), row in by_question.items():
            if baseline != 'full':
                continue
            other = by_question[qid, view]
            assert row['visual_pairs_received'] == other['visual_pairs_received']
            spans, surviving = indexes(row['bank_status'])
            other_spans, other_surviving = indexes(other['bank_status'])
            stats['different_bank_structure'] += spans != other_spans
            stats['different_surviving_pair_ids'] += surviving != other_surviving
            diagnostics.append(dict(id=qid, role=row['role'], view=view, full_structure=spans,
                pruned_structure=other_spans, full_surviving_pairs=sorted(surviving),
                pruned_surviving_pairs=sorted(other_surviving)))
        counts[view] = dict(stats)
    result = dict(pairs_per_view=len(full_events), questions_per_view=18, views=counts,
        explanation='Surviving coordinates track provenance, not semantic evidence preservation.',
        sources_sha256={n: hashlib.sha256((OLD / n).read_bytes()).hexdigest()
                        for n in ('answers.jsonl', 'event_trace.jsonl')}, details=diagnostics)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'pilot_coverage_audit.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    main()
