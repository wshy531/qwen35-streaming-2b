"""Choose the cost panel using question type and prefix length only."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/stream_prune_cost_prepare_20261102_01'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    questions = [q for q in read('runs/evidence_scale_prepare_20261102_01/panel.json') if q['dataset'] == 'StreamingBench']
    selected, used = [], set()
    for kind in sorted({q['type'] for q in questions}):
        ordered = sorted([q for q in questions if q['type'] == kind], key=lambda q: (q['query_time_s'], q['id']))
        for name, half in [('shorter_half', ordered[:len(ordered)//2]), ('longer_half', ordered[len(ordered)//2:])]:
            median = half[len(half)//2]['query_time_s']
            candidates = sorted(half, key=lambda q: (abs(q['query_time_s']-median), hashlib.sha256(q['id'].encode()).hexdigest()))
            chosen = next(q for q in candidates if q['source_id'] not in used)
            used.add(chosen['source_id'])
            selected.append(dict(chosen, selection_stratum=name))
    assert len(selected) == len(used) == 20
    blocks = [[], [], []]
    loads = [0.0, 0.0, 0.0]
    for q in sorted(selected, key=lambda q: (-q['query_time_s'], q['id'])):
        index = min(range(3), key=lambda i: loads[i])
        blocks[index].append(q['id'])
        loads[index] += q['query_time_s']
    arms = ['full', 'fixed12', 'heuristic11']
    jobs = []
    for i, block in enumerate(blocks):
        for arm in arms[i:] + arms[:i]:
            jobs.append(dict(block=i, arm=arm, question_ids=block))
    save('panel.json', selected)
    save('jobs.json', jobs)
    # Old answers are read only after the panel has been determined and written.
    reference = []
    selected_ids = {q['id'] for q in selected}
    for line in (ROOT/'runs/evidence-scale-stream-tail-220170/answers.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['id'] in selected_ids and row['arm'] in arms:
            reference.append({k: row[k] for k in ['id', 'arm', 'generated_ids', 'input_ids_sha256', 'position_ids_sha256', 'visual_tokens', 'visible_pairs', 'source_state']})
    assert len(reference) == 60
    save('reference.json', reference)
    warmup = dict(min(selected, key=lambda q: (q['query_time_s'], q['id'])))
    warmup['query_time_s'] = min(15., warmup['query_time_s'])
    save('warmup.json', warmup)
    pairs = sum(next(r['source_state']['complete_pairs'] for r in reference if r['id'] == q['id']) for q in selected)
    protocol = dict(questions=20, sources=20, types=dict(collections.Counter(q['type'] for q in selected)),
                    selection='Per type choose the median prefix of each duration half; unique files; no outputs used',
                    prefix_min_s=min(q['query_time_s'] for q in selected), prefix_max_s=max(q['query_time_s'] for q in selected),
                    prefix_sum_s=sum(q['query_time_s'] for q in selected), block_prefix_sum_s=loads,
                    arms=arms, model_loads=9, formal_answers=60, warmup_answers=9, max_generated_tokens=69*128,
                    formal_visual_calls=pairs*3, warmup_visual_call_upper=72, max_visual_calls=pairs*3+72,
                    dtype='FP32', canvas_hw=[288,384], sampling='causal 1FPS, V0 auto4', bank_pairs=8,
                    processing='Encode every committed pair from start until query; one method per process',
                    quality_reference='Exact generated IDs, input IDs, positions, pair indices and source state from frozen 150-question run',
                    thresholds=dict(min_pipeline_reduction_fraction=.05, max_option_loss_on_panel=0,
                                    interpretation='Engineering screen, not statistical noninferiority. One run per method and source; no tuning on this panel.'),
                    budget=dict(GPU='one A800', partition='g078t', comment='gx_llm', cpus=4, memory_GiB=24, minutes=90,
                                output_MiB=64, API=0, training=0, downloads=0, requeue=False),
                    scope='Synchronized causal fast replay on A800. No 1x or Orin speed claim; disk cache uncontrolled and method order rotated.',
                    prior_scores_used_for_selection=False)
    save('protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == '__main__':
    main()
