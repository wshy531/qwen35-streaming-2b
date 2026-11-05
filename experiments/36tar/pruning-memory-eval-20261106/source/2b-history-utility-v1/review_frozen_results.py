"""CPU-only audit of frozen head outputs; no fitting or checkpoint selection."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/history_utility_review_20261105_01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    choices = [c for n in range(5) for c in itertools.combinations(range(4), n)]
    results, hashes = {}, {}
    for cohort, folder, expected in (
        ('N271_development', 'history-utility-fit-220511', 271),
        ('N345_short', 'history-utility-short-heads-220532', 345),
    ):
        results[cohort] = {}
        for head in ('content_visible', 'blind_history'):
            path = ROOT / 'runs' / folder / f'{head}_paired.jsonl'
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            index = {(r['id'], r['shift'], r['view']): r for r in rows}
            assert len(index) == len(rows) == expected * 2 * 7
            ids = sorted({r['id'] for r in rows})
            assert len(ids) == expected
            selections, per_question, paired = Counter(), [], {}
            for qid in ids:
                h = [index[qid, s, 'head'] for s in (0, 2)]
                assert h[0]['subset'] == h[1]['subset']
                selections[str(choices[h[0]['subset']])] += 1
                p = {}
                for view in ('head', 'recent', 'all', 'matched_count_nearest', 'same_head_binary'):
                    v = [index[qid, s, view] for s in (0, 2)]
                    p[view] = sum(r['gold_probability'] for r in v) / 2
                per_question.append(p)
            for base in ('recent', 'all', 'matched_count_nearest', 'same_head_binary'):
                gain = loss = probability_up_answer_down = 0
                for qid in ids:
                    for shift in (0, 2):
                        h, b = index[qid, shift, 'head'], index[qid, shift, base]
                        gain += h['correct'] and not b['correct']
                        loss += b['correct'] and not h['correct']
                        probability_up_answer_down += (h['gold_probability'] > b['gold_probability'] + 1e-7
                                                       and b['correct'] and not h['correct'])
                differences = [p['head'] - p[base] for p in per_question]
                paired[base] = dict(gain=gain, loss=loss,
                    probability_better_questions=sum(x > 1e-6 for x in differences),
                    probability_worse_questions=sum(x < -1e-6 for x in differences),
                    probability_equal_questions=sum(abs(x) <= 1e-6 for x in differences),
                    mean_probability_difference=sum(differences) / expected,
                    probability_up_but_answer_wrong_rows=probability_up_answer_down)
            result = dict(questions=expected, selections=dict(selections), paired=paired)
            log = ROOT / 'runs' / folder / f'{head}_training.json'
            if log.exists():
                hashes[str(log.relative_to(ROOT))] = hashlib.sha256(log.read_bytes()).hexdigest()
                history = read(log)
                result['training'] = dict(first=history[0], final=history[-1],
                    relative_online_loss_reduction=1-history[-1]['training_huber']/history[0]['training_huber'],
                    note='online minibatch training averages, not frozen-checkpoint downstream evaluation')
            results[cohort][head] = result
    OUT.mkdir(exist_ok=True)
    payload = dict(results=results, input_sha256=hashes, fitting=False, GPU=False)
    (OUT / 'analysis.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
