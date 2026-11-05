"""Score the frozen four-arm run offline with the original content parser."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-sember-v1'))
from sember_common import read, save
from sember_protocol import parse_choice
import json

ARMS = ('guarded', 'old_CE', 'initial', 'fixed12')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    receipt = read(run / 'completed.json')
    assert receipt['status'] == 'completed' and receipt['answers'] == 1912
    label_path = ROOT / 'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'
    req_path = ROOT / 'runs/sember_main_prepare_20261025_01/inputs/requests.json'
    labels = {r['question_id']: r for r in read(label_path)}
    questions = {r['question_id']: r for r in read(req_path)}
    scores = {arm: {} for arm in ARMS}
    for line in (run / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        qid, arm = row['question_id'], row['arm']
        assert qid not in scores[arm] and row['video_id'] == questions[qid]['video_id']
        choice = parse_choice(row['text'], len(questions[qid]['options']))
        scores[arm][qid] = choice == labels[qid]['correct_letter']
    assert all(rows.keys() == questions.keys() for rows in scores.values())
    panels = {
        'all': list(questions),
        'gap_ge120s': [q for q in questions if labels[q]['end_gap_proxy_s'] >= 120],
        'gap_ge300s': [q for q in questions if labels[q]['end_gap_proxy_s'] >= 300],
    }
    output = {'scores': {}, 'guarded_vs_each': {}}
    for arm, rows in scores.items():
        output['scores'][arm] = {name: {'correct': sum(rows[q] for q in ids), 'total': len(ids)}
                                 for name, ids in panels.items()}
        if arm != 'guarded':
            counts = Counter((rows[q], scores['guarded'][q]) for q in questions)
            output['guarded_vs_each'][arm] = {'gain': counts[False, True], 'loss': counts[True, False]}
    save(run / 'analysis.json', output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
