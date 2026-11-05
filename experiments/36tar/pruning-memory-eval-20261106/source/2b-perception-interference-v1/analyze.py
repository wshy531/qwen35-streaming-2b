"""Frozen content analysis; no method ranking from three opened sources."""
import argparse
from pathlib import Path
from common import ROOT, PREP, INPUTS, OLD, read, save
from scoring import scored, apply_development_review, summarize


def analyze_records(raw, tasks, schedule, review, references):
    byid = {t['id']: t for t in tasks}
    original = [scored(byid[r['task_id']], r['arm'], r['shift'], r['text']) for r in raw]
    reviewed = [apply_development_review(row, review) for row in original]
    primary = summarize(reviewed, tasks, schedule)
    reference_only = summarize(original, tasks, schedule)
    bykey = {(r['task_id'], r['arm'], r['shift']): r for r in original}
    reproduction = []
    for ref in references:
        row = bykey[(ref['task_id'], ref['arm'], ref['shift'])]
        reproduction.append(dict(**ref, new_choice_original=row['content_choice_original'],
            matched=row['content_choice_original'] == ref['expected_content_choice_original']))
    patterns = []
    for task in tasks:
        if task['role'] != 'current':
            continue
        row = dict(task_id=task['id'], harmful_history_shifts=[], same_length_swap_recovery_shifts=[],
                   full_swap_choice_changes=[])
        for shift in range(5):
            a, b, c = [bykey[(task['id'], arm, shift)] for arm in ('recent', 'full', 'history_value_swap')]
            if a['content_correct'] and not b['content_correct']:
                row['harmful_history_shifts'].append(shift)
                if c['content_correct']:
                    row['same_length_swap_recovery_shifts'].append(shift)
            if b['content_choice_original'] != c['content_choice_original']:
                row['full_swap_choice_changes'].append(shift)
        patterns.append(row)
    reproduced = all(r['matched'] for r in reproduction) and len(reproduction) == 30
    signal = any(len(r['same_length_swap_recovery_shifts']) >= 2 for r in patterns)
    decision = ('stop_baseline_reproduction_mismatch' if not reproduced else
                'one_conditional_reader_candidate_warranted_for_preparation' if signal else
                'end_this_diagnostic_without_adding_a_reader_candidate')
    return dict(primary=primary, unchanged_official_label_reference=reference_only,
        old_reference_reproduction=dict(matched=sum(r['matched'] for r in reproduction),
            total=len(reproduction), passed=reproduced, rows=reproduction),
        current_readout_patterns=patterns, decision=decision, scored_rows=reviewed,
        no_automatic_new_GPU=True, independent_generalization_claim=False,
        semantic_overlap_and_feature_distribution_not_fully_controlled=True,
        claim_scope='same-model development readout diagnosis, not general perception or long-memory accuracy')


def main(run):
    import json
    done = read(run / 'completed.json')
    assert done['status'] == 'complete' and done['formal_answers'] == 90 and done['visual_calls'] == 367
    raw = [json.loads(line) for line in (run / 'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    result = analyze_records(raw, read(INPUTS / 'tasks_LOCAL_ONLY.json'), read(INPUTS / 'schedule.json'),
        read(PREP / 'assistant_review_LOCAL_ONLY.json'), read(PREP / 'old_reference_predictions_LOCAL_ONLY.json'))
    save(run / 'analysis_LOCAL_ONLY.json', result)
    print(dict(primary_response_count=sum(r['denominator'] for r in result['primary']['totals'].values()),
        baseline_reproduction=result['old_reference_reproduction']['passed'], decision=result['decision']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
