"""Score only complete, verified 300-question runs. Labels stay local."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/2b-evidence-eval-v1'))
from response_v4 import parse

OUT = ROOT/'runs/ls_prune_expansion_prepare_20261103_01'
VIEWS = ('full','fixed12','heuristic11','mask_ce')
CAP = ('R4H4','R4H8','R8H4','R8H8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def score(run):
    done = read(run/'completed.json')
    assert (done['main'],done['capacity'],done['replayed'],done['visual_calls']) == (1200,216,108,108879)
    questions = read(OUT/'regression300.json')
    diagnostic = read(ROOT/'runs/midprune_memory_prepare_20261102_01/questions.json')
    labels = read(ROOT/'runs/evidence_scale_prepare_20261102_01/labels_LOCAL_ONLY.json')
    old_labels = read(ROOT/'runs/midprune_memory_prepare_20261102_01/labels_LOCAL_ONLY.json')
    labels.update({key: item['correct_index'] for key,item in old_labels.items()})
    ids = {q['id']:q for q in questions+diagnostic}
    rows = {}
    for line in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = row['id'],row['arm']
        assert key not in rows and row['branch_released'] and row['read_state_unchanged']
        q = ids[row['id']]
        assert all(n['end'] <= q['query_time_s']+1e-6 for n in row['nodes'])
        row.update(parse(row['text'],len(q['options']),'B'))
        selected = row['original_selected_index']
        if row['answer_conflict']:
            selected = None
        row['correct'] = None if selected is None else selected==labels[row['id']]
        rows[key] = row
    assert len(rows)==1416

    def group(qs,arms):
        scores, comparisons = {}, {}
        for arm in arms:
            subset = [rows[q['id'],arm] for q in qs]
            counts = Counter(row['correct'] for row in subset)
            scores[arm] = dict(questions=len(qs), correct=counts[True], incorrect=counts[False],
                unresolved=counts[None], mean_visual_tokens=sum(r['visual_tokens'] for r in subset)/len(subset))
        for a,b in itertools.combinations(arms,2):
            pairs = [(rows[q['id'],a]['correct'],rows[q['id'],b]['correct']) for q in qs]
            comparisons[a+'__'+b] = dict(a_only_correct=pairs.count((True,False)),
                b_only_correct=pairs.count((False,True)), both_correct=pairs.count((True,True)),
                both_incorrect=pairs.count((False,False)), unresolved=sum(None in x for x in pairs))
        return dict(scores=scores,comparisons=comparisons)

    main = {}
    for ds in ('NExT-QA','StreamingBench'):
        qs = [q for q in questions if q['dataset']==ds]
        assert len(qs)==150
        main[ds] = group(qs,VIEWS)
        main[ds]['by_type'] = {t:group([q for q in qs if q['type']==t],VIEWS)
                               for t in sorted({q['type'] for q in qs})}
    capacity = {v: {role:group([q for q in diagnostic if q['role']==role],
                      [v+'_'+x for x in CAP]) for role in ('current','old')} for v in VIEWS[1:]}
    with (run/'analysis_LOCAL_ONLY.json').open('x',encoding='utf-8') as f:
        json.dump(dict(main=main,capacity=capacity,
            scoring='Unresolved answers await content review and are counted separately from incorrect.'),f,indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    score(parser.parse_args().run)
