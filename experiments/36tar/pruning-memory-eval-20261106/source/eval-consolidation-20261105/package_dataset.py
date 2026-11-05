"""Build one question/label registry for the consolidated evaluation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from build import ROOT

DEST = ROOT / 'datasets/streaming-evaluation-v1'
SECOND = ROOT / 'runs/eval_consolidation_20261105_02'
THIRD = ROOT / 'runs/eval_consolidation_20261105_03'
sys.path.insert(0, str(ROOT / 'experiments/2b-evidence-eval-v1'))
from response import question


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def canonical(row, dataset, role, labels, media_map):
    original = row['options']
    labelled = [re.match(r'^\s*' + chr(65 + i) + r'[.)]\s+(.+)$', value)
                for i, value in enumerate(original)]
    options = [m.group(1).strip() for m in labelled] if all(labelled) else original[:]
    assert 2 <= len(options) <= 5 and len(set(options)) == len(options)
    assert all(v.strip() for v in options)
    qid = row['id']
    label = labels.get(qid)
    assert label is None or isinstance(label, int) and 0 <= label < len(options)
    source = row.get('source_id', row.get('source_name'))
    data = dict(id=qid, dataset=dataset, role=role, source_id=source,
                question=row['question'], options=options,
                query_time_s=float(row['query_time_s']), type=row.get('type', row.get('role', 'unknown')),
                memory_requirement=row.get('memory_requirement', 'unreviewed'),
                original_options=original, answer_key_id=qid)
    if row.get('length_bin'):
        data['length_bin'] = row['length_bin']
    if row.get('duration_s'):
        data['duration_s'] = row['duration_s']
        assert data['query_time_s'] <= data['duration_s'] + 1e-6
    if row.get('source_group'):
        data['source_group'] = row['source_group']
    if row.get('case_id'):
        data.update(case_id=row['case_id'], stage=row['stage'], diagnostic_role=row['role'])
    if row.get('media_path'):
        data.update(media_path=row['media_path'], media_sha256=row['media_sha256'])
    else:
        binding = media_map[source if source in media_map else source + '.mp4']
        data.update(media_path=binding['source_path'], media_sha256=binding['media_sha256'],
                    media_binding='legacy_D18_readonly_root', sampled_trace_binding=source)
    if row.get('evidence_bounds_s'):
        data['evidence_bounds_s'] = row['evidence_bounds_s']
        data['recent_input_pts'] = row['recent_input_pts']
    prompt = question(data, 0, 'B')
    assert not re.search(r'(?m)^[A-E]\.\s+[A-E]\.\s+', prompt)
    data['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    return data


def main():
    DEST.mkdir(exist_ok=True)
    dev = read(SECOND / 'development.json')
    dev_labels = read(SECOND / 'development_labels_LOCAL_ONLY.json')
    short = read(THIRD / 'short_validation_questions.json')
    short_labels = read(THIRD / 'short_validation_labels_LOCAL_ONLY.json')
    regression = set(read(SECOND / 'regression_ids.json'))
    bindings = {r['name']: r for r in read(ROOT / 'runs/midprune_memory_prepare_20261102_01/sources.json')}
    sources, suite_rows, labels = [], {}, {}
    for key, records, group_labels, role in [
        ('development', dev, dev_labels, 'opened_development'),
        ('short_validation', short, short_labels, 'project_source_disjoint_validation')]:
        rows = [canonical(q, q['dataset'], role, group_labels, bindings) for q in records]
        labels.update(group_labels)
        suite_rows[key] = rows
    suite_rows['regression'] = [q for q in suite_rows['development'] if q['id'] in regression]
    suite_rows['development_streamingbench'] = [q for q in suite_rows['development'] if q['dataset'] == 'StreamingBench']
    suite_rows['development_short'] = [q for q in suite_rows['development'] if q['dataset'] == 'NExT-QA']
    suite_rows['development_diagnosis'] = [q for q in suite_rows['development'] if q['dataset'] == 'S-EMBER']
    stream = read(THIRD / 'stream_main_technical_candidates.json')
    stream_labels = read(ROOT / 'runs/eval_consolidation_20261105_01/new_streamingbench_labels_LOCAL_ONLY.json')
    suite_rows['streamingbench_candidate'] = [
        canonical(q, 'StreamingBench', 'pending_capture_isolation', stream_labels, bindings) for q in stream]
    labels.update({q['id']: stream_labels[q['id']] for q in stream})
    draft_path = THIRD / 'diagnosis_prefix_review/paired_draft_questions.json'
    drafts = read(draft_path)
    draft_labels = {k:ord(v)-65 for k,v in read(draft_path.with_name('paired_draft_labels_LOCAL_ONLY.json')).items()}
    suite_rows['diagnosis_candidate'] = [
        canonical(q, 'StreamingBench', 'assistant_reviewed_draft', draft_labels, bindings) for q in drafts]
    labels.update(draft_labels)
    assert len(suite_rows['development']) == 228 and len(suite_rows['regression']) == 153
    assert len(short) == len({q['source_group'] for q in short}) == 60
    assert {q['source_group'] for q in short}.isdisjoint(
        q['source_group'] for q in dev if q['dataset']=='NExT-QA')
    all_records = {}
    registry = {}
    for name, rows in suite_rows.items():
        assert len(rows) == len({q['id'] for q in rows})
        path = DEST / f'{name}.json'
        save(path, rows)
        registry[name] = dict(path=path.name, questions=len(rows),
            source_files=len({q['source_id'] for q in rows}),
            dataset_counts=dict(Counter(q['dataset'] for q in rows)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            runnable=name not in ('streamingbench_candidate','diagnosis_candidate'),
            score_groups_separately=name in ('development','regression'))
        for q in rows:
            if q['id'] in all_records:
                assert all_records[q['id']] == q
            all_records[q['id']] = q
    assert set(all_records) == set(labels)
    save(DEST / 'answer_keys_LOCAL_ONLY.json', labels)
    legacy_protocol = read(ROOT / 'runs/midprune_memory_prepare_20261102_01/protocol.json')
    save(DEST / 'bindings_LOCAL_ONLY.json', dict(
        project_root='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar',
        legacy_D18_readonly_root=legacy_protocol['media_root'],
        legacy_D18_trace='runs/midprune_memory_prepare_20261102_01/sources.json'))
    protocol = dict(version='20261105-v1', prompt='B', option_shifts_primary=[0],
        generation_max_new_tokens=128, parser='response_v4_first_choice',
        sampling=dict(StreamingBench='V0Pairs causal PTS 1 FPS', NExT_QA='existing 2 FPS EOF pair protocol',
                      legacy_D18='existing fixed sampled_trace, preserve original 429-pair replay'),
        canvas=[288,384], interpolation='existing two-stage BICUBIC letterbox',
        recent_baseline_real_pairs=8, full_tokens_per_pair=108, pruned_tokens_per_pair=54,
        LS_visual_token_cap=864, current_history_labels='only explicit reviewed items',
        incomplete_answers='fail completeness check; report unresolved separately',
        evidence_judgement='auxiliary, separate from option correctness; fee authorization required',
        source_identity_limit='short uses mapped VidOR source and exact SHA; StreamingBench parent identities pending')
    save(DEST / 'protocol.json', protocol)
    save(DEST / 'registry.json', dict(version='20261105-v1', suites=registry,
        labels_path='answer_keys_LOCAL_ONLY.json', unique_questions=len(all_records),
        bindings_path='bindings_LOCAL_ONLY.json',
        progress_record='E:/VIT Model/实验总纲.md'))
    print(json.dumps({k:v['questions'] for k,v in registry.items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
