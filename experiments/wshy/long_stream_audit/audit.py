"""Recompute paired A06 changes; never infer missing labels or admission groups."""
import argparse
import importlib.util
import json
from pathlib import Path

from experiments.wshy.online_memory_2b.frozen import PACKAGE
from experiments.wshy.online_memory_2b.isolation import output_path
from experiments.wshy.online_memory_2b.protocol import read, save, sha

ARMS = ('M0', 'M2', 'A06')
SCORING = PACKAGE / 'bundle/experiments/2b-short-event-v1/scoring.py'


def one(records, kind):
    matches = [r for r in records if r.get('kind') == kind]
    if len(matches) != 1:
        raise ValueError('exactly one captured record required: ' + kind)
    return matches[0]


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def validate_grid(capture):
    grid = one(capture, 'answer_grid')
    rows = grid['rows']
    ids = [r['id'] for r in rows]
    if grid['queries'] != 34 or len(rows) != 34 or len(set(ids)) != 34:
        raise ValueError('complete unique 34-question comparison required')
    if (sum(q.startswith('ovo:') for q in ids) != 28
            or sum(q.startswith('local_longstream_v1:') for q in ids) != 6):
        raise ValueError('original28 and manual6 must remain separate')
    for row in rows:
        if set(row['answers']) != set(ARMS) or set(row['generated_ids']) != set(ARMS):
            raise ValueError('complete M0/M2/A06 grid required')
        for arm in ARMS:
            tokens = row['generated_ids'][arm]
            if (not isinstance(row['answers'][arm], str) or not isinstance(tokens, list)
                    or not 1 <= len(tokens) <= 8
                    or any(type(t) is not int or t < 0 for t in tokens)):
                raise ValueError('invalid answer or generated token budget')
    comparisons = [r for r in capture if r.get('kind') == 'comparison']
    if len(comparisons) != 2 or {r['reference'] for r in comparisons} != {'M0', 'M2'}:
        raise ValueError('both baseline replay receipts required')
    for receipt in comparisons:
        reference = receipt['reference']
        changed = {r['id'] for r in rows
                   if r['generated_ids'][reference] != r['generated_ids']['A06']}
        claimed = receipt['A06_changed_ids']
        if (receipt['compared'] != 34 or receipt['baseline_replay_count'] != 34
                or receipt['baseline_id_mismatches'] or receipt['baseline_frame_mismatches']
                or len(set(claimed)) != len(claimed) or set(claimed) != changed):
            raise ValueError('baseline replay or paired-change receipt mismatch')
    return sorted(rows, key=lambda r: r['id'])


def verified_labels(capture, requests, rows):
    allowed = {r['id'] for r in rows}
    matched = one(capture, 'upstream_matching_questions')['records']
    truths = {}
    for item in matched:
        qid, raw = item['request_id'], item['record']
        if qid not in allowed or qid in truths or qid not in requests:
            raise ValueError('duplicate or unbound original label')
        request = requests[qid]
        # This audit consumes previously retrieved request metadata, not a new
        # independent reading of the cluster's original requests.json.
        if (qid != 'ovo:' + str(raw['id']) or raw['video'] != request['source_path']
                or raw['realtime'] != request['query_time_s']
                or raw['question'] != request['question'] or raw['options'] != request['options']):
            raise ValueError('original label/request identity mismatch: ' + qid)
        index = raw['gt']
        if (type(index) is not int or not 0 <= index < len(raw['options'])
                or raw['answer'] != raw['options'][index]):
            raise ValueError('invalid original label')
        truths[qid] = index
    return truths


def scoring_module():
    spec = importlib.util.spec_from_file_location('_long_stream_frozen_scoring', SCORING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(capture, labels, request_audit):
    rows = validate_grid(capture)
    changed_metadata = request_audit['changed_cases']
    metadata = {r['id']: r for r in changed_metadata}
    changed = {r['id'] for r in rows if r['generated_ids']['M2'] != r['generated_ids']['A06']}
    if len(metadata) != len(changed_metadata) or set(metadata) != changed:
        raise ValueError('changed request metadata must match the recomputed M2/A06 changes')
    for row in rows:
        if row['id'] in metadata and any(
                metadata[row['id']][arm]['text'] != row['answers'][arm] for arm in ('M2', 'A06')):
            raise ValueError('request audit answer drift')
    truths = verified_labels(labels, metadata, rows)
    scoring = scoring_module()
    details, review = [], []
    for row in rows:
        qid = row['id']
        meta = metadata.get(qid, {})
        group = 'original28' if qid.startswith('ovo:') else 'manual6'
        arms = {}
        for arm in ARMS:
            measure = scoring.measure(row['answers'][arm], len(meta.get('options', [0]*4)),
                                      truths.get(qid))
            # None can also mean an unparseable answer. Missing labels must
            # never accidentally score True through None == None.
            if qid not in truths:
                measure['strict_correct'] = measure['content_correct'] = None
            arms[arm] = dict(text=row['answers'][arm], generated_ids=row['generated_ids'][arm],
                             **measure)
        details.append(dict(id=qid, cohort=group, arms=arms,
                            original_label_verified=qid in truths,
                            original_label_index=truths.get(qid),
                            evidence_admission='pending', human_semantic_review='pending'))
        # The review queue deliberately contains no labels, answers or gains.
        review.append(dict(id=qid, cohort=group,
                           question_metadata_status='available' if meta else 'pending',
                           question=meta.get('question'), options=meta.get('options'),
                           source_path=meta.get('source_path'), query_time_s=meta.get('query_time_s'),
                           admission_group='pending', reviewer=None, reviewed_at=None,
                           evidence_intervals=None, sampled_evidence_present=None,
                           later_repetition_or_scene_cue=None, long_memory_necessary=None,
                           manual_label=None, status='pending'))
    changed_details = [r for r in details if r['id'] in changed]
    gains = [r['id'] for r in changed_details if r['original_label_verified']
             and not r['arms']['M2']['content_correct'] and r['arms']['A06']['content_correct']]
    losses = [r['id'] for r in changed_details if r['original_label_verified']
              and r['arms']['M2']['content_correct'] and not r['arms']['A06']['content_correct']]
    summary = dict(schema='long-stream-paired-audit-v1', source_job=217786,
                   scope='offline validation of previously retrieved comparison captures',
                   questions=34, answers=102, cohorts=dict(original28=28, manual6=6),
                   baseline_replay_receipts_valid=True,
                   M2_A06_changed_ids=sorted(changed),
                   verified_original_labels=len(truths),
                   changed_original_label_corrections=gains,
                   changed_original_label_regressions=losses,
                   original_label_retrieval_selection='changed cases only; biased subset',
                   overall_accuracy=None, supported16_accuracy=None, manual6_accuracy=None,
                   formal_scoring_status='blocked_by_missing_admission_and_labels',
                   expected_admission_counts=dict(supported_original=16, original_ambiguous=11,
                                                  original_not_established=1, manual=6),
                   format_counts={arm: dict(strict=sum(r['arms'][arm]['format_compliant'] for r in details),
                                            leading=sum(r['arms'][arm]['content_parseable'] for r in details))
                                  for arm in ARMS},
                   independent_holdout=False, new_cluster_jobs=0, new_model_loads=0)
    return summary, details, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers-capture', type=Path, required=True)
    parser.add_argument('--labels-capture', type=Path, required=True)
    parser.add_argument('--request-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inputs = {key: output_path(getattr(args, key)) for key in
              ('answers_capture', 'labels_capture', 'request_audit')}
    output = output_path(args.out)
    if output.exists():
        raise ValueError('new audit directory required')
    summary, details, review = build(records(inputs['answers_capture']), records(inputs['labels_capture']),
                                     read(inputs['request_audit']))
    summary['input_receipt'] = {key: dict(path=str(path), sha256=sha(path)) for key, path in inputs.items()}
    summary['code_receipt'] = dict(audit_sha256=sha(Path(__file__)), frozen_scoring_sha256=sha(SCORING))
    output.mkdir(parents=True, exist_ok=False)
    save(output/'summary.json', summary)
    save(output/'paired_details.json', details)
    save(output/'review_without_predictions.json', review)
    print(json.dumps({k: summary[k] for k in ('questions', 'M2_A06_changed_ids',
                      'verified_original_labels', 'overall_accuracy', 'formal_scoring_status')}))


if __name__ == '__main__':
    main()
