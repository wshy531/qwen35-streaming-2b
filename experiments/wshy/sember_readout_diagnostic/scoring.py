"""Content correctness and paired changes; no answer-format metric."""
from collections import Counter
from ..long_video_gpu_2b.scoring import scorer
from .plan import ARMS


def build(queries, labels, predictions, historical):
    qmap = {q['id']: q for q in queries}
    gold = {r['id']: r for r in labels}
    indexed = {(r['id'], r['arm']): r for r in predictions}
    expected = {(qid, arm) for qid in qmap for arm in ARMS}
    if (len(queries) != len(qmap) or len(labels) != len(gold) or set(gold) != set(qmap)
            or len(indexed) != len(predictions) or set(indexed) != expected):
        raise ValueError('incomplete or duplicate comparison grid')
    measure = scorer().measure
    details, values = [], {}
    for qid, q in qmap.items():
        label = gold[qid]
        if (type(label['correct_index']) is not int or not 0 <= label['correct_index'] < 5
                or label['correct_letter'] != 'ABCDE'[label['correct_index']]):
            raise ValueError('gold letter/index mismatch')
        for arm in ARMS:
            row = indexed[qid, arm]
            if row['source_path'] != q['source_path'] or row['query_time_s'] != q['query_time_s']:
                raise ValueError('prediction source/query mismatch')
            if row['status'] not in ('ok', 'no_committed_evidence_sample'):
                raise ValueError('failed run is not a complete result')
            if row['status'] != 'ok' and arm not in ('native_evidence', 'current_evidence'):
                raise ValueError('only evidence arms may have a missing-sample status')
            measured = measure(row['text'], len(q['options']), label['correct_index'])
            # Unparsed outputs require review; never call them semantically wrong.
            correct = measured['content_correct'] if row['status'] == 'ok' and measured['content_parseable'] else None
            values[qid, arm] = correct
            details.append(dict(id=qid, arm=arm, status=row['status'],
                content_choice=measured['content_choice'], content_correct=correct,
                semantic_review='pending', category=label['question_category']))
    def paired(a, b):
        eligible = [qid for qid in qmap if values.get((qid, a)) is not None and values.get((qid, b)) is not None]
        gains = [qid for qid in eligible if values[qid, a] and not values[qid, b]]
        losses = [qid for qid in eligible if not values[qid, a] and values[qid, b]]
        return dict(evaluable_questions=len(eligible), corrections=gains, new_losses=losses, net=len(gains)-len(losses))
    old_text = [r for r in historical if r['arm'] == 'text_only']
    if len(old_text) != len(qmap) or {r['id'] for r in old_text} != set(qmap):
        raise ValueError('historical text baseline incomplete')
    for row in old_text:
        qid = row['id']
        if (row['status'] != 'ok' or row['source_path'] != qmap[qid]['source_path']
                or row['query_time_s'] != qmap[qid]['query_time_s']):
            raise ValueError('historical text input identity mismatch')
        m = measure(row['text'], 5, gold[qid]['correct_index'])
        values[qid, 'text_only_legacy'] = m['content_correct'] if m['content_parseable'] else None
    totals = {arm: dict(questions=len(qmap), content_correct=sum(values[qid, arm] is True for qid in qmap),
        unresolved=sum(values[qid, arm] is None for qid in qmap)) for arm in (*ARMS, 'text_only_legacy')}
    pairs = {a+'_vs_'+b: paired(a, b) for a, b in (
        ('text_only_clean', 'text_only_legacy'), ('native_evidence', 'current_evidence'),
        ('native_evidence', 'native_prefix'), ('current_evidence', 'text_only_clean'),
        ('native_prefix', 'text_only_clean'))}
    return dict(totals=totals, paired=pairs, human_semantic_review='pending',
        gold_letter_counts=dict(Counter(r['correct_letter'] for r in labels)),
        scope='102 development questions; author evidence oracle is offline, not a deployable streaming method',
        caution='native/current comparison includes model implementation, timestamp layout, positions and prefill path'), details
