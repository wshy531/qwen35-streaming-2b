"""Offline publication integrity and score aggregation; no semantic auto-judge."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    manifest = read('manifest.json')
    for name, expected in manifest.items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, 'SHA mismatch: '+name)
    data = read('scores.json')
    rows = data['评分明细']
    require(len(rows) == 800 and len({r['review_id'] for r in rows}) == 800, '800 unique records')
    raw = [json.loads(s) for s in (ROOT/'answers.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    require(len(raw) == 800, '800 original answers')
    lookup = {(r['question_id'],r['arm']):r for r in raw}
    require(len(lookup) == 800, 'unique raw keys')
    for r in rows:
        require(r['评分'] in (None,0,1,2) and bool(r['依据']), 'score/rationale')
        require(lookup[(r['题目ID'],r['模型'])]['text'] == r['模型回答'], 'answer identity')
    pending = sorted({r['题号'] for r in rows if r['评分'] is None})
    require(pending == data['pending_questions'] and len(pending) == 11, 'pending questions')
    require(sum(r['评分'] is None for r in rows) == 31, 'pending answers')
    changes = [r for r in rows if r['评分'] != r['上版评分']]
    require(len(changes) == 19, 'change count')
    require({r['review_id'] for r in changes} == {r['review_id'] for r in read('score_changes.json')}, 'change identity')
    by_arm = defaultdict(list)
    for r in rows: by_arm[r['模型']].append(r)
    require(set(by_arm) == {'native_prefix','LS','A06_full','text_only'}, 'four arms')
    output = {}
    for arm, all_rows in by_arm.items():
        require(len(all_rows) == 200 and {r['题号'] for r in all_rows} == set(range(1,201)), '200 per arm')
        items = [r for r in all_rows if r['题号'] not in pending]
        counts = Counter(r['评分'] for r in items)
        points = sum(r['评分'] for r in items)
        stored = data['summary'][arm]
        require(len(items) == stored['common_questions'] == 189, 'common denominator')
        require(points == stored['points'] and stored['max_points'] == 378, 'points')
        for score in range(3): require(counts[score] == stored['score_'+str(score)], 'counts')
        require(abs(100*points/378 - stored['normalized_score_pct']) < 1e-10, 'percentage')
        require(abs(100*counts[2]/189 - stored['full_correct_pct']) < 1e-10, 'correct percentage')
        output[arm] = {'points':points,'denominator':378,'score_counts':dict(counts)}
    provenance = read('provenance.json')
    require(hashlib.sha256((ROOT/'answers.jsonl').read_bytes()).hexdigest() == provenance['historical_completion']['answers_sha256'], 'historical raw answer SHA')
    require(all(r['stop_reason'] == 'eos' for r in raw), 'EOS completion')
    print(json.dumps({'verified':True,'answers':800,'common_questions':189,'pending_answers':31,'summary':output},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
