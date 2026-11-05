"""State-change, base-readability and history effects; rotations are not samples."""
from collections import defaultdict
import re


def parse_original(text, shift):
    match = re.search(r'\b([A-E])\b', text.strip().upper())
    return None if match is None else (ord(match.group(1))-65+shift) % 5


def analyze(raw, tasks, cases, schedule):
    expected = {(r['task_id'],r['arm'],r['shift']) for r in schedule}
    rows = {(r['task_id'],r['arm'],r['shift']):r for r in raw}
    if len(rows) != len(raw) or set(rows) != expected:
        raise ValueError('missing, duplicate or foreign responses')
    byid = {t['id']:t for t in tasks}
    scored = {}
    totals = defaultdict(lambda:dict(correct=0, denominator=0, unparsed=0, format_only=0))
    for key, row in rows.items():
        tid, arm, shift = key
        task = byid[tid]
        choice = parse_original(row['text'], shift)
        eligible = not (task['role']=='old' and arm=='history_value_swap')
        correct = choice==task['correct_index'] if eligible else None
        scored[key] = dict(task_id=tid, arm=arm, shift=shift, content_choice=choice,
            content_correct=correct, content_scored=eligible)
        group = task['stage']+'/'+task['role']+'/'+arm
        totals[group]['unparsed'] += int(choice is None)
        totals[group]['format_only'] += int(row['text'].strip() in 'ABCDE' and len(row['text'].strip())==1)
        if eligible:
            totals[group]['denominator'] += 1
            totals[group]['correct'] += int(correct)
    patterns = []
    support_sources = set()
    for case in cases:
        cid = case['id']
        pattern = dict(case_id=cid, source_path=case['source_path'],
            foundation_unreadable_shifts=[], history_added_errors=[], history_added_corrections=[],
            stale_answer_shifts=[], matched_recovery_shifts=[], both_current_correct={}, old_and_late_correct={})
        for arm in ('recent','full','history_value_swap'):
            pattern['both_current_correct'][arm] = sum(
                bool(scored[(cid+':early:current',arm,s)]['content_correct'])
                and bool(scored[(cid+':late:current',arm,s)]['content_correct']) for s in range(5))
        for arm in ('recent','full'):
            pattern['old_and_late_correct'][arm] = sum(
                bool(scored[(cid+':late:old',arm,s)]['content_correct'])
                and bool(scored[(cid+':late:current',arm,s)]['content_correct']) for s in range(5))
        for shift in range(5):
            early = scored[(cid+':early:current','recent',shift)]
            a,b,c = [scored[(cid+':late:current',arm,shift)] for arm in ('recent','full','history_value_swap')]
            readable = bool(early['content_correct']) and bool(a['content_correct'])
            if not readable:
                pattern['foundation_unreadable_shifts'].append(shift)
            if a['content_correct'] and not b['content_correct']:
                pattern['history_added_errors'].append(shift)
            if not a['content_correct'] and b['content_correct']:
                pattern['history_added_corrections'].append(shift)
            stale = readable and not b['content_correct'] and b['content_choice']==case['answers'][0]
            if stale:
                pattern['stale_answer_shifts'].append(shift)
                if c['content_correct']:
                    pattern['matched_recovery_shifts'].append(shift)
        if len(pattern['matched_recovery_shifts'])>=2:
            support_sources.add(case['source_path'])
        patterns.append(pattern)
    return dict(totals=dict(totals), patterns=patterns, support_source_count=len(support_sources),
        prepare_one_reader_candidate=len(support_sources)>=2,
        stop_if_no_signal=True, all_foundation_failures_retained=True,
        semantic_case_count=len(cases), source_count=len({c['source_path'] for c in cases}),
        raw_answers=len(raw), scored_answers=sum(g['denominator'] for g in totals.values()),
        independent_unit='source, not question time or option rotation',
        scope='opened development diagnosis; not action meaning or formal long-memory evaluation')
