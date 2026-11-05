"""CPU regression for the delegated review and uncertain-label scoring overlay."""
from copy import deepcopy
from common import ROOT, INPUTS, PREP, read, save, sha, require_admission, require_development_review
from scoring import scored, apply_development_review, summarize


def rejected(function):
    try:
        function()
    except ValueError:
        return True
    raise AssertionError('invalid review was accepted')


def main():
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    review = read(PREP / 'assistant_review_LOCAL_ONLY.json')
    tasks = read(INPUTS / 'tasks_LOCAL_ONLY.json')
    schedule = read(INPUTS / 'schedule.json')
    assert require_development_review(manifest, review)
    failures = 0
    for mutation in ('independent', 'formal', 'wrong_key', 'uncertain_primary', 'bad_SHA', 'missing_row', 'duplicate_row'):
        bad = deepcopy(review)
        if mutation == 'independent': bad['independent_review'] = True
        elif mutation == 'formal': bad['formal_benchmark_admitted'] = True
        elif mutation == 'wrong_key': bad['tasks'][0]['reviewed_correct_index'] = 4
        elif mutation == 'uncertain_primary': bad['tasks'][4]['primary_content_scored'] = True
        elif mutation == 'bad_SHA': bad['manifest_sha256'] = '0' * 64
        elif mutation == 'missing_row': bad['tasks'].pop()
        elif mutation == 'duplicate_row': bad['tasks'].append(bad['tasks'][0])
        assert rejected(lambda: require_development_review(manifest, bad))
        failures += 1
    independent = read(PREP / 'independent_review_LOCAL_ONLY.json')
    assert rejected(lambda: require_admission(manifest, independent))
    byid = {t['id']: t for t in tasks}
    rows, original = [], []
    for item in schedule:
        task = byid[item['task_id']]
        text = chr(65 + (task['correct_index'] - item['shift']) % 5) + ' because of the observed object.'
        raw = scored(task, item['arm'], item['shift'], text)
        original.append(raw)
        rows.append(apply_development_review(raw, review))
    result = summarize(rows, tasks, schedule)
    assert sum(r['denominator'] for r in result['totals'].values()) == 65
    assert sum(r['denominator'] for r in summarize(original, tasks, schedule)['totals'].values()) == 75
    assert result['both_questions_correct'] == {'recent': 10, 'full': 10}
    assert result['pair_rotation_denominators'] == {'recent': 10, 'full': 10}
    for row in rows:
        if row['task_id'] == 'P03:old':
            assert row['content_correct'] is None and row['content_scored'] is False
    for row in result['per_semantic_question_changes']:
        if row['task_id'] == 'P03:old':
            assert row['corrections'] is None and row['new_errors'] is None
    proof = dict(status='passed_CPU_review_and_scoring_only', negative_review_checks=failures,
        development_review_accepted=True, independent_human_gate_still_not_satisfied=True,
        primary_response_count=65, unchanged_official_reference_count=75, raw_response_count=90,
        pair_rotation_denominators=result['pair_rotation_denominators'],
        all_raw_responses_retained=True, GPU_jobs=0, pretrained_loads=0,
        human_review_wait_removed_for_development=True, capability_claim=False,
        source_sha256={str(p.relative_to(ROOT).as_posix()): sha(p) for p in (
            ROOT / 'experiments/2b-perception-interference-v1/common.py',
            ROOT / 'experiments/2b-perception-interference-v1/scoring.py',
            ROOT / 'experiments/2b-perception-interference-v1/review_content.py',
            ROOT / 'experiments/2b-perception-interference-v1/check_review.py')})
    save(PREP / 'assistant_review_CPU_check.json', proof)
    print(proof)


if __name__ == '__main__':
    main()
