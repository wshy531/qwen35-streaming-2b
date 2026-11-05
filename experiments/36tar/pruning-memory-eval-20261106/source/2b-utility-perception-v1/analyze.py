"""Score content, including same-time current and old pairs, on complete runs."""
import json
import re
import sys
from pathlib import Path

from evaluate import ARMS, PREP, read, save


def main(out):
    done = read(out/'completed.json')
    assert (done['answers'], done['reference_matches'], done['updates']) == (2000,640,0)
    tasks = read(PREP/'requests.json')
    labels = read(PREP/'labels_LOCAL_ONLY.json')
    correct = {}
    with (out/'answers.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['task_id'], row['arm'], row['shift']
            assert key not in correct and row['branch_released'] and row['bank_unchanged']
            match = re.search(r'\b([A-E])\b', row['text'].strip().upper())
            choice = (ord(match[1])-65+row['shift'])%5 if match else None
            correct[key] = choice == labels[row['task_id']]
    assert set(correct)=={(t['id'],a,s) for t in tasks for a in ARMS for s in range(5)}
    groups = {'all': tasks, 'current': [t for t in tasks if t['role']=='current'],
              'old': [t for t in tasks if t['role']=='old'],
              'official': [t for t in tasks if t['role']=='official_reference'],
              'regression64': [t for t in tasks if t['evaluation_group']=='regression64'],
              'long36': [t for t in tasks if t['evaluation_group']!='regression64']}
    pairs = []
    for old in groups['old']:
        current, = [t for t in groups['current'] if t['case_id']==old['case_id'] and t['stage']=='late']
        assert old['query_time_s']==current['query_time_s'] and old['source_name']==current['source_name']
        pairs.append((current['id'],old['id']))
    summaries = {}
    for group, items in groups.items():
        def score(arm):
            return [correct[(t['id'],arm,s)] for t in items for s in range(5)]
        summaries[group] = {}
        for arm in ARMS:
            scores = score(arm)
            contrasts = {}
            for base in ('recent','full'):
                paired = list(zip(scores,score(base),strict=True))
                contrasts[base] = dict(gain=paired.count((True,False)), loss=paired.count((False,True)))
            summaries[group][arm] = dict(correct=sum(scores), total=len(scores), contrasts=contrasts)
    same_time = {}
    for arm in ARMS:
        both = sum(correct[(c,arm,s)] and correct[(h,arm,s)] for c,h in pairs for s in range(5))
        same_time[arm] = dict(total=5*len(pairs), correct=both)
    summary = dict(groups=summaries, same_time_both=same_time)
    save(out/'analysis.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':
    main(Path(sys.argv[1]))
