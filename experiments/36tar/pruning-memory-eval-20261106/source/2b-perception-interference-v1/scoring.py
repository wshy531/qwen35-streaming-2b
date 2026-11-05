"""Content, formatting and evidence substitution are distinct measurements."""
from collections import defaultdict
import sys
from common import ROOT, ARMS
sys.path.insert(0, str(ROOT / 'experiments/2b-sember-v1'))
from sember_protocol import parse_choice, format_ok


def scored(task, arm, shift, text):
    letter = parse_choice(text, 5)
    original = None if letter is None else (ord(letter) - 65 + shift) % 5
    eligible = not (task['role'] == 'old' and arm == 'history_value_swap')
    return dict(task_id=task['id'], pair_id=task['pair_id'], role=task['role'], arm=arm,
        shift=shift, content_choice_original=original, content_scored=eligible,
        content_correct=(original == task['correct_index']) if eligible else None,
        parseable=letter is not None, format_only=format_ok(text, 5))


def apply_development_review(row, review):
    """Preserve original-label reference; do not turn uncertain labels into truth."""
    decision = next(r for r in review['tasks'] if r['task_id'] == row['task_id'])
    result = dict(row, original_label_reference_correct=row['content_correct'],
                  review_scope=review['scope'], review_decision=decision['decision'])
    if decision['decision'] == 'uncertain_reference_only':
        result.update(content_correct=None, content_scored=False)
    return result


def summarize(rows, tasks, schedule):
    expected = {(r['task_id'], r['arm'], r['shift']) for r in schedule}
    bykey = {(r['task_id'], r['arm'], r['shift']): r for r in rows}
    if len(bykey) != len(rows) or set(bykey) != expected:
        raise ValueError('missing, duplicate or foreign answers')
    totals = defaultdict(lambda: dict(correct=0, denominator=0, parseable=0, format_only=0))
    for row in rows:
        group = totals[row['role'] + '/' + row['arm']]
        group['parseable'] += int(row['parseable'])
        group['format_only'] += int(row['format_only'])
        if row['content_scored']:
            group['denominator'] += 1
            group['correct'] += int(row['content_correct'])
        elif row['content_correct'] is not None:
            raise ValueError('do not score replaced historical evidence against original truth')
    paired, pair_denominators = {}, {}
    for arm in ('recent', 'full'):
        eligible = [(bykey[(pair + ':old', arm, shift)], bykey[(pair + ':current', arm, shift)])
                    for pair in sorted({t['pair_id'] for t in tasks}) for shift in range(5)
                    if bykey[(pair + ':old', arm, shift)]['content_scored']
                    and bykey[(pair + ':current', arm, shift)]['content_scored']]
        paired[arm] = sum(bool(a['content_correct']) and bool(b['content_correct']) for a, b in eligible)
        pair_denominators[arm] = len(eligible)
    changes = []
    for task in tasks:
        for left, right in (('recent', 'full'), ('full', 'history_value_swap')):
            a = [bykey[(task['id'], left, s)] for s in range(5)]
            b = [bykey[(task['id'], right, s)] for s in range(5)]
            score_eligible = all(x['content_scored'] and y['content_scored'] for x, y in zip(a, b))
            changes.append(dict(task_id=task['id'], comparison=left + '->' + right,
                changed_answers=sum(x['content_choice_original'] != y['content_choice_original']
                                    for x, y in zip(a, b)),
                corrections=None if not score_eligible else
                    sum(not x['content_correct'] and bool(y['content_correct']) for x, y in zip(a, b)),
                new_errors=None if not score_eligible else
                    sum(bool(x['content_correct']) and not y['content_correct'] for x, y in zip(a, b))))
    return dict(totals=dict(totals), both_questions_correct=paired,
        pair_rotation_denominator=next(iter(pair_denominators.values()))
            if len(set(pair_denominators.values())) == 1 else None,
        pair_rotation_denominators=pair_denominators, per_semantic_question_changes=changes,
        independent_unit='source/pair; rotations are correlated repeated measurements',
        interpretation='opened development diagnosis only; no accuracy includes formatting')
