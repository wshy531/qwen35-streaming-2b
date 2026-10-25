"""Offline original-label scoring, with paired changes and evidence proxies."""
from collections import Counter, defaultdict

from ..online_memory_2b.isolation import output_path
from ..online_memory_2b.protocol import read, rows, save, sha
from ..long_video_gpu_2b.scoring import scorer
from .inputs import FIXED, base_inputs, bundle_path, counts
from .gates import verify_cpu


def evidence_positions(sources, labels):
    gold = {row['id']: row for row in labels}
    positions = {}
    for item in sources:
        for query in item['queries']:
            label = gold[query['id']]
            prefix = item['expected_prefix'][query['id']]
            frames = item['frames'][:prefix['committed_frames']]
            hits = [i for i, frame in enumerate(frames)
                    if label['answer_start_time'] <= frame['pts_s'] <= label['answer_end_time']]
            recent = [i for i in hits if i >= max(0, len(frames)-16)]
            group = ('no_committed_sample_hit' if not hits else
                     'outside_M0_only' if not recent else
                     'inside_M0_only' if len(recent) == len(hits) else 'cross_window')
            positions[query['id']] = group
    return positions


def build(queries, labels, predictions, measure, positions=None):
    byid, gold = {q['id']: q for q in queries}, {r['id']: r for r in labels}
    indexed = {(p['id'], p['arm']): p for p in predictions}
    expected = {(qid, arm) for qid in byid for arm in FIXED['arms']}
    if (len(queries) != 102 or len(byid) != 102 or len(gold) != len(labels)
            or set(gold) != set(byid) or len(predictions) != 612 or set(indexed) != expected):
        raise ValueError('complete unique 102-question/six-arm grid required')
    groups = defaultdict(list)
    detail = []
    for qid, query in byid.items():
        label = gold[qid]
        truth = label['correct_index']
        if (type(truth) is not int or not 0 <= truth < 5
                or label['correct_letter'] != 'ABCDE'[truth]
                or query['options'][truth][:2] != label['correct_letter'] + '.'
                or label['video_id'] != query['source_path'].rsplit('/', 1)[-1][:-4]
                or label['query_time_s'] != query['query_time_s']
                or not 0 <= label['answer_start_time'] <= label['answer_end_time'] <= query['query_time_s']):
            raise ValueError('official label identity or temporal order changed')
        gap = query['query_time_s'] - label['answer_end_time']
        age = '0_to_16s' if gap < 16 else '16_to_60s' if gap < 60 else '60_to_120s' if gap < 120 else 'at_least_120s'
        for name in ('all_original102', 'category/' + label['question_category'], 'author_gap/' + age):
            groups[name].append(qid)
        if positions is not None:
            groups['sampled_author_position/' + positions[qid]].append(qid)
        for arm in FIXED['arms']:
            answer = indexed[qid, arm]
            if (answer.get('status') != 'ok' or not isinstance(answer.get('text'), str)
                    or answer.get('source_path') != query['source_path']
                    or answer.get('query_time_s') != query['query_time_s']):
                raise ValueError('failed answer or request identity drift')
            measured = measure(answer['text'], len(query['options']), truth)
            detail.append(dict(id=qid, arm=arm, category=label['question_category'],
                source_path=query['source_path'], author_gap_s=gap,
                independent_semantic_review='pending',
                content_choice=measured['content_choice'],
                content_correct=measured['content_correct'],
                content_parseable=measured['content_parseable']))
    values = {(row['id'], row['arm']): row for row in detail}
    tables, paired = {}, {}
    for group, ids in groups.items():
        tables[group] = {arm: dict(questions=len(ids),
            content_correct=sum(values[qid, arm]['content_correct'] for qid in ids))
            for arm in FIXED['arms']}
        paired[group] = {}
        for reference in ('M0', 'M2', 'M2_A06'):
            paired[group][reference] = {}
            for arm in FIXED['arms']:
                if arm == reference:
                    continue
                gains = [qid for qid in ids if not values[qid, reference]['content_correct']
                         and values[qid, arm]['content_correct']]
                losses = [qid for qid in ids if values[qid, reference]['content_correct']
                          and not values[qid, arm]['content_correct']]
                paired[group][reference][arm] = dict(
                    corrections=gains, new_losses=losses, net=len(gains)-len(losses))
    summary = dict(status='official_label_development_comparison', unique_official_questions=102,
        sources=18, main_answers=612, tables=tables, paired_comparisons=paired,
        gold_letter_counts=dict(Counter(row['correct_letter'] for row in labels)),
        parsed_letter_counts={arm: dict(Counter('ABCDE'[values[qid, arm]['content_choice']]
            if values[qid, arm]['content_choice'] is not None else 'unparsed' for qid in byid))
            for arm in FIXED['arms']},
        independent_holdout=False, human_semantic_review='pending',
        old_evidence_necessity='pending; author interval age/coverage is a proxy',
        question_suffix_tokens=512, previous_entry_suffix_tokens=128,
        interpretation='unambiguous option-content correctness against original official labels; no output-format metric and no human semantic judgment',
        stable_ability_improvement_established=False)
    return summary, detail


def score(run):
    protocol, sources, proof = verify_cpu(run)
    run = output_path(run)
    complete = read(run / 'gpu/completed.json')
    hashes = {name: sha(run / 'gpu' / name) for name in ('predictions.jsonl', 'events.jsonl')}
    expected_pairs = counts(sources)['encoded_pairs']
    if (complete.get('status') != 'complete' or complete.get('pretrained_loads') != 1
            or complete.get('main_answers') != 612 or complete.get('visual_forward_calls') != expected_pairs
            or complete.get('source_pairs_encoded') != expected_pairs or complete.get('reset_checks') != 90
            or complete.get('protocol_sha256') != sha(run / 'protocol.json')
            or complete.get('CPU_preflight_sha256') != sha(run / 'cpu/preflight.json')
            or complete.get('output_sha256') != hashes):
        raise ValueError('complete hash-bound 102-question GPU result required')
    queries = base_inputs()[0]
    labels = read(bundle_path() / 'labels-offline-only.json')
    summary, detail = build(queries, labels, rows(run / 'gpu/predictions.jsonl'), scorer().measure,
                            evidence_positions(sources, labels))
    summary.update(protocol_sha256=sha(run / 'protocol.json'), raw_output_hashes=hashes,
                   completed_sha256=sha(run / 'gpu/completed.json'))
    out = run / 'score'
    out.mkdir(exist_ok=False)
    save(out / 'summary.json', summary)
    save(out / 'details.json', detail)
    save(out / 'review-pending.json', [dict(id=q['id'], reviewer=None, review_status='pending',
        long_memory_necessary=None, later_repetition_or_context_cues=None,
        semantic_answer_judgment=None) for q in queries])
