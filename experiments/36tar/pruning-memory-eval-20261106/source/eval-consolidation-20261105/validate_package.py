"""Validate the registered manifests and label isolation without model inference."""
import hashlib
import json
from pathlib import Path
import re

from package_dataset import DEST, ROOT, read, save
from dataset_cli import score


def main():
    registry = read(DEST / 'registry.json')
    keys = read(DEST / registry['labels_path'])
    suites, ids = {}, set()
    for name, item in registry['suites'].items():
        path = DEST / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        rows = read(path)
        assert len(rows) == item['questions'] == len({q['id'] for q in rows})
        assert len({q['source_id'] for q in rows}) == item['source_files']
        for q in rows:
            assert 'answer' not in q and 'gold' not in q and 'label' not in q
            assert q['id'] in keys and 0 <= keys[q['id']] < len(q['options'])
            assert q['query_time_s'] >= 0 and q['media_sha256']
            assert len(q['options']) == len(set(q['options']))
            if 'duration_s' in q:
                assert q['query_time_s'] <= q['duration_s'] + 1e-6
        suites[name] = rows
        ids.update(q['id'] for q in rows)
    assert ids == set(keys) and len(ids) == registry['unique_questions']
    development = {q['id']: q for q in suites['development']}
    assert all(development[q['id']] == q for q in suites['regression'])
    assert set(development) == {q['id'] for name in ('development_streamingbench',
            'development_short', 'development_diagnosis') for q in suites[name]}
    # Different views can assign different IDs to the same source/time/question.
    # Record that overlap instead of calling all registered IDs independent items.
    records = {q['id']:q for rows in suites.values() for q in rows}
    presentations = {}
    clean = lambda value: re.sub(r'\s+', ' ', value).strip().casefold()
    for qid, q in records.items():
        identity = [q['media_sha256'], round(q['query_time_s'],6), clean(q['question']),
                    [clean(x) for x in q['options']]]
        digest = hashlib.sha256(json.dumps(identity,ensure_ascii=False).encode()).hexdigest()
        presentations.setdefault(digest, []).append(qid)
    duplicates = [sorted(group) for group in presentations.values() if len(group)>1]
    assert all(len({keys[qid] for qid in group}) == 1 for group in duplicates)
    identity_index = dict(unique_ids=len(ids), distinct_exact_presentations=len(presentations),
                          duplicate_id_groups=duplicates,
                          independence_not_implied=True,
                          identity_fields='media SHA, query time, normalized question and ordered options')
    save(DEST / 'question_identity_index.json', identity_index)
    registry.update(unique_question_ids=len(ids), distinct_exact_presentations=len(presentations),
                    unique_questions_field_semantics='legacy field counts IDs, not independent annotations',
                    identity_index_path='question_identity_index.json')
    save(DEST / 'registry.json', registry)
    # Exercise the newly registered answer-index conversion and primary scoring.
    panel = suites.get('diagnosis_anchors', suites['short_validation'])
    synthetic = [dict(id=q['id'], text='Answer: ' + chr(65 + keys[q['id']]) + '\nEvidence: schema check only.')
                 for q in panel]
    result = score(panel, keys, synthetic)
    assert sum(v['correct'] for v in result['groups'].values()) == len(panel)
    assert all(v['unresolved'] == 0 for v in result['groups'].values())
    checks = {}
    for name, answers in [('missing', synthetic[:-1]), ('duplicate', synthetic + synthetic[:1]),
                          ('unknown', synthetic + [dict(id='not-a-question', text='Answer: A')])]:
        try:
            score(panel, keys, answers)
        except ValueError:
            checks[name] = 'rejected'
        else:
            raise AssertionError(name)
    record = dict(suites={name:len(rows) for name,rows in suites.items()},
                  unique_ids=len(ids), scoring_schema_rows=len(panel),
                  distinct_exact_presentations=len(presentations),
                  duplicated_presentation_groups=len(duplicates),
                  invalid_answer_sets=checks, model_inference=False,
                  note='Synthetic schema checks are not capability results.')
    save(DEST / 'validation_receipt.json', record)
    files = ['experiments/streamingbench-router-v1/v0_source.py',
             'experiments/streamingbench-interface-v1/source.py',
             'experiments/2b-evidence-eval-v1/response.py',
             'experiments/2b-evidence-eval-v1/response_v4.py']
    template = dict(dataset_registry_sha256=hashlib.sha256((DEST / 'registry.json').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((DEST / 'protocol.json').read_bytes()).hexdigest(),
        implementations={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        run_fields_required=['suite', 'method_id', 'checkpoint_sha256', 'backbone_revision',
            'precision', 'write_policy', 'read_policy', 'event_policy', 'visual_token_budget',
            'recent_real_pairs', 'device', 'answer_generation_or_option_classification'],
        primary_shifts=[0], optional_stability_shifts='report separately',
        pure_text_control='same question/options/template; no visual content',
        full_history_reference='all causal frames through query; not a bounded deployment method',
        model_result=None)
    save(DEST / 'execution_contract_template.json', template)
    print(record)


if __name__ == '__main__':
    main()
