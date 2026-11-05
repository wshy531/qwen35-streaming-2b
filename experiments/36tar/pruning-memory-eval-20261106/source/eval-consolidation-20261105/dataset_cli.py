"""Export frozen evaluation inputs and score complete answer sets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from package_dataset import ROOT, DEST, read

sys.path.insert(0, str(ROOT / 'experiments/2b-evidence-eval-v1'))
from response import question
from response_v4 import parse


def write(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def score(rows, labels, answers):
    expected = {q['id']:q for q in rows}
    given = {}
    for r in answers:
        qid = r['id']
        if qid in given or qid not in expected or r.get('shift',0) != 0:
            raise ValueError('Unknown or repeated question, or unsupported option shift')
        q = expected[qid]
        value = parse(r['text'], len(q['options']), 'B')
        choice = None if value['answer_conflict'] else value['original_selected_index']
        given[qid] = dict(id=qid, dataset=q['dataset'], source_id=q['source_id'],
            selected_index=choice, correct=None if choice is None else choice == labels[qid],
            explanation_present=value['explanation_present'], conflict=value['answer_conflict'])
    if given.keys() != expected.keys():
        raise ValueError(f'Incomplete answer set: {len(given)} received, {len(expected)} required')
    by_dataset = defaultdict(list)
    for qid, result in given.items():
        by_dataset[expected[qid]['dataset']].append(result)
    scores = {}
    for dataset, rs in by_dataset.items():
        correct = sum(r['correct'] is True for r in rs)
        unresolved = sum(r['correct'] is None for r in rs)
        scores[dataset] = dict(correct=correct, resolved_wrong=len(rs)-correct-unresolved,
            unresolved=unresolved, total=len(rs), sources=len({r['source_id'] for r in rs}),
            option_correct_fraction=correct/len(rs) if not unresolved else None)
    return dict(groups=scores, per_question=list(given.values()),
                score_protocol='B / response_v4 / original option order',
                unresolved_policy='review separately before reporting a content-correct rate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('inspect','export','score'))
    p.add_argument('--suite', required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--answers', type=Path)
    a = p.parse_args()
    registry = read(DEST/'registry.json')
    if a.suite not in registry['suites']:
        p.error('Choose a suite listed in registry.json')
    entry = registry['suites'][a.suite]
    path = DEST / entry['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Question manifest changed after registry creation')
    rows = read(path)
    if a.command == 'inspect':
        print(json.dumps(entry,ensure_ascii=False,indent=2))
        return
    if not a.out:
        p.error('--out is required')
    if not entry['runnable']:
        p.error('This suite still has admission prerequisites')
    if a.command == 'export':
        bindings = read(DEST / registry['bindings_path'])
        exported = []
        for q in rows:
            legacy = q.get('media_binding') == 'legacy_D18_readonly_root'
            media = (bindings['legacy_D18_readonly_root'] + '/' + q['source_id'] + '.mp4'
                     if legacy else bindings['project_root'] + '/' + q['media_path'])
            exported.append(dict(id=q['id'],dataset=q['dataset'],source_id=q['source_id'],
                query_time_s=q['query_time_s'],media_path=media,
                media_sha256=q['media_sha256'],
                sampled_trace_path=bindings['legacy_D18_trace'] if legacy else None,
                prompt=question(q,0,'B'),prompt_sha256=q['prompt_sha256']))
        write(a.out,exported)
        return
    if not a.answers:
        p.error('--answers is required')
    labels = read(DEST / registry['labels_path'])
    with a.answers.open(encoding='utf-8-sig') as f:
        answers = [json.loads(line) for line in f if line.strip()]
    write(a.out,score(rows,labels,answers))


if __name__=='__main__':
    main()
