"""Audit existing opened development data. No torch, media I/O or network."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
from contract import source_groups, supervision, runtime_request, normalized_question

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'runs/recent_evidence_prepare_20261028_01/inputs'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(input_dir, output_dir):
    tasks = read(input_dir / 'evaluation_tasks_LOCAL_ONLY.json')
    requests = read(input_dir / 'requests.json')
    rows = read(input_dir / 'frozen_branch_outputs_LOCAL_ONLY.json')
    sources = {s['name']: s for s in read(input_dir / 'sources.json')}
    by_task = defaultdict(list)
    for row in rows:
        by_task[row['task_id']].append(row)
    by_id = {r['id']: r for r in requests}
    task_ids = {t['id'] for t in tasks}
    if len(task_ids) != len(tasks) or len(by_id) != len(requests) or task_ids != set(by_id) or task_ids != set(by_task):
        raise ValueError('task/request/output IDs must match uniquely')
    groups = source_groups(requests, sources)
    labels = []
    for task in tasks:
        r = by_id[task['id']]
        if task['source_name'] != r['source_name'] or task['query_time_s'] != r['query_time_s']:
            raise ValueError('source/query mismatch')
        if len(r['recent_frames']) != 8 or any(f['pts_s'] > r['query_time_s'] for f in r['recent_frames']):
            raise ValueError('eight causal recent frames required')
        labels.append(dict(task_id=task['id'], **groups[task['id']], **supervision(by_task[task['id']]),
                           cohort=task['cohort'], role=task['role'],
                           source_name=task['source_name'], case_id=task.get('case_id'),
                           independent_content_review=task['independent_content_review']))
    summaries = []
    gates = {}
    for held in range(3):
        for side in ('fit', 'heldout'):
            subset = [r for r in labels if (r['fold'] == held) == (side == 'heldout')]
            positive = [r for r in subset if r['target_gain'] > 0]
            negative = [r for r in subset if r['target_gain'] < 0]
            summaries.append(dict(heldout_fold=held, side=side, tasks=len(subset),
                groups=len({r['leakage_group'] for r in subset}),
                fit_weight_tasks=sum(r['fit_weight'] for r in subset),
                positive_tasks=len(positive), negative_tasks=len(negative),
                positive_groups=len({r['leakage_group'] for r in positive}),
                negative_groups=len({r['leakage_group'] for r in negative})))
            required_tasks = 6 if side == 'fit' else 2
            required_groups = 3 if side == 'fit' else 2
            gates[f'fold{held}_{side}_two_sided_coverage'] = (
                len(positive) >= required_tasks and len(negative) >= required_tasks
                and len({r['leakage_group'] for r in positive}) >= required_groups
                and len({r['leakage_group'] for r in negative}) >= required_groups)
    # Independent content supervision and a new-source test are separately absent.
    gates['independent_sufficiency_labels_available'] = False
    gates['new_unopened_sources_frozen'] = False
    summary = dict(tasks=len(tasks), rotation_rows=len(rows), rotations_per_task=5,
        media_files=len(sources), parent_ids=len({r['parent_id'] for r in labels}),
        leakage_groups=len({r['leakage_group'] for r in labels}),
        preference_counts=dict(Counter(r['preference'] for r in labels)),
        zero_weight_both_wrong_tasks=sum(r['fit_weight'] == 0 for r in labels),
        opposing_rotation_preference_tasks=sum(r['opposing_rotation_preferences'] for r in labels),
        totals=dict(sum_counters(labels)), folds=summaries, gates=gates,
        retrospective_coverage_ready=all(v for k, v in gates.items() if 'coverage' in k),
        formal_sufficiency_training_ready=False, real_feature_extraction_done=False,
        real_head_training_done=False, new_GPU_jobs=0, new_model_answers=0,
        split_status='retrospective_grouped_CV_all_sources_previously_opened',
        group_limit='parent ID/media bytes/question identity; shooting-session independence not established',
        split_does_not_depend_on_outcomes=True,
        all_rows_retained_for_evaluation=True, unknown_support_not_relabeled=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    save(output_dir / 'policy_targets_LOCAL_ONLY.json', labels)
    save(output_dir / 'feature_requests.json', [runtime_request(r) for r in requests])
    save(output_dir / 'media_resolution_LOCAL_ONLY.json',
         {r['id']: dict(source_name=r['source_name'], sha256=sources[r['source_name']]['sha256']) for r in requests})
    save(output_dir / 'audit.json', summary)
    save(output_dir / 'source_lock.json', [dict(path=str(p.relative_to(ROOT)), sha256=digest(p), bytes=p.stat().st_size)
         for p in sorted(input_dir.glob('*.json'))])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def sum_counters(labels):
    c = Counter()
    for r in labels:
        for k in ('recent_correct','full_correct','recent_only','full_only','both_correct','both_wrong'):
            c[k] += r[k]
    return c

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=INPUT)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.input, a.output)
