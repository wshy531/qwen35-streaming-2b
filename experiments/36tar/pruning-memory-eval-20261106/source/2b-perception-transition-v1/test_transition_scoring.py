"""CPU synthetic contracts; these results are not model capability evidence."""
from copy import deepcopy
from transition_common import INPUTS, read
from transition_scoring import analyze


def main():
    cases = read(INPUTS / 'cases_LOCAL_ONLY.json')
    tasks = read(INPUTS / 'tasks_LOCAL_ONLY.json')
    schedule = read(INPUTS / 'schedule.json')
    byid = {r['id']: r for r in tasks}
    raw = [dict(task_id=r['task_id'], arm=r['arm'], shift=r['shift'],
                text=chr(65+(byid[r['task_id']]['correct_index']-r['shift']) % 5))
           for r in schedule]
    def run(rows):
        return analyze(rows, tasks, cases, schedule)
    baseline = run(raw)
    assert baseline['raw_answers'] == 270 and baseline['scored_answers'] == 240
    assert not baseline['prepare_one_reader_candidate']
    assert all(g['correct'] == g['denominator'] for g in baseline['totals'].values())
    assert sum(g['denominator'] for k,g in baseline['totals'].items() if k.endswith('/old/history_value_swap')) == 0
    checks = ['all_correct_does_not_activate_candidate', 'old_intervention_unscored', '270_raw_240_content']

    def stale(rows, case_ids):
        values = deepcopy(rows)
        for row in values:
            cid = row['task_id'].split(':')[0]
            if cid in case_ids and row['task_id'].endswith(':late:current') and row['arm'] == 'full':
                case = next(c for c in cases if c['id'] == cid)
                row['text'] = chr(65+(case['answers'][0]-row['shift']) % 5)
        return values
    same_source = run(stale(raw, ['T01', 'T06']))
    assert same_source['support_source_count'] == 1 and not same_source['prepare_one_reader_candidate']
    two_source_rows = stale(raw, ['T01', 'T02'])
    two_source = run(two_source_rows)
    assert two_source['support_source_count'] == 2 and two_source['prepare_one_reader_candidate']
    checks += ['same_video_two_cases_count_as_one_source', 'two_source_matched_recovery_activates_only_development_gate']
    unreadable = deepcopy(two_source_rows)
    for row in unreadable:
        if row['task_id'].endswith(':early:current') and row['arm'] == 'recent':
            row['text'] = chr(65+(byid[row['task_id']]['correct_index']+1-row['shift']) % 5)
    foundation = run(unreadable)
    assert not foundation['prepare_one_reader_candidate']
    assert all(len(p['foundation_unreadable_shifts']) == 5 for p in foundation['patterns'])
    assert all(not p['stale_answer_shifts'] for p in foundation['patterns'])
    checks.append('foundation_failure_not_misattributed_to_history')
    explained = deepcopy(raw)
    for row in explained:
        row['text'] += '. This is the object visible in the current pair.'
    content = run(explained)
    assert all(g['correct'] == g['denominator'] and g['format_only'] == 0 for g in content['totals'].values())
    checks.append('content_correct_with_explanation_still_correct_format_reported_separately')
    for invalid in (raw[:-1], raw+[raw[0]], [dict(raw[0], task_id='foreign')]+raw[1:]):
        try:
            run(invalid)
        except ValueError:
            continue
        raise AssertionError('invalid response set accepted')
    checks.append('missing_duplicate_foreign_responses_rejected')
    return dict(status='passed_CPU_synthetic_scoring', checks=checks, check_count=len(checks),
                model_answers=0, pretrained_loads=0, GPU_jobs=0)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from transition_common import save
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = main()
    save(args.output, result)
    print(result)
