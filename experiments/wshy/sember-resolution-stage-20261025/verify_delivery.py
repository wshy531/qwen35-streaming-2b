"""Read-only standard-library verification. Official labels are optional/external."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
from statistics import median

BUNDLE=Path(__file__).resolve().parent
REPO=BUNDLE.parents[2]
RESOLUTIONS={'low':48,'mid':108,'high':192}
CONDITIONS=[(a,r) for a in ('LS','A06','native') for r in RESOLUTIONS]+[('text_only','none')]


def safe(root,name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe relative path')
    result=root/path
    if not result.resolve().is_relative_to(root.resolve()):raise ValueError('path escape')
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
def key(row):return row['id'],row['arm'],row['resolution'],row['shift']


def grid(rows,ids):
    indexed={key(r):r for r in rows}
    expected={(qid,a,r,s) for qid in ids for a,r in CONDITIONS for s in range(5)}
    if len(rows)!=len(indexed) or set(indexed)!=expected:raise ValueError('duplicate or incomplete 5100-answer grid')
    return indexed


def replay_equal(actual,reference):
    for name in ('generated_ids','text','input_ids_sha256','position_ids_sha256','frames_sha256','pixels_sha256'):
        if actual[name]!=reference[name]:raise ValueError('historical replay drift: '+name)


def verify(labels=None):
    for name,row in read(BUNDLE/'DELIVERY_MANIFEST.json')['files'].items():
        path=safe(BUNDLE,name)
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('publication content changed: '+name)
    manifest=read(BUNDLE/'EVIDENCE_MANIFEST.json')
    for name,row in manifest['files'].items():
        p=safe(BUNDLE,name)
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['stored_sha256']:raise ValueError('evidence hash mismatch: '+name)
        if 'source_sha256' in row and sha(p)!=row['source_sha256']:raise ValueError('original evidence bytes changed')
    bindings=read(BUNDLE/'SOURCE_BINDINGS.json')['executed_runtime']
    if len(bindings)!=226:raise ValueError('execution source coverage changed')
    for name,digest in bindings.items():
        if sha(safe(REPO,name))!=digest:raise ValueError('executed source changed: '+name)
    base=BUNDLE/'evidence/job218458'
    done=read(base/'gpu/completed.json');cpu=read(base/'cpu/passed.json');protocol=read(base/'protocol.json')
    if done['status']!='complete' or done['protocol_sha256']!=sha(base/'protocol.json'):raise ValueError('completion/protocol mismatch')
    if done['CPU_proof_sha256']!=sha(base/'cpu/passed.json') or cpu['protocol_sha256']!=sha(base/'protocol.json'):raise ValueError('CPU proof mismatch')
    if cpu['status']!='passed' or cpu['pretrained_loads']!=0 or len(cpu['tiny_answers'])!=9:raise ValueError('CPU proof incomplete')
    for name,digest in cpu['hashes'].items():
        if sha(safe(base/'cpu',name))!=digest:raise ValueError('CPU proof artifact mismatch')
    counts=re.findall(r'Ran (\d+) tests? in ',(base/'cpu/contracts.txt').read_text(encoding='utf-8'))
    if list(map(int,counts))!=[11,17,13,20,11]:raise ValueError('72 test denominator changed')
    if re.search(r'FAILED|ERROR|skipped=',(base/'cpu/contracts.txt').read_text(encoding='utf-8')):raise ValueError('CPU test failure or skip')
    fresh=lines(base/'gpu/predictions.jsonl');reused=read(base/'historical_reuse.json')
    if sha(base/'gpu/predictions.jsonl')!=done['predictions_sha256'] or sha(base/'gpu/events.jsonl')!=done['events_sha256']:raise ValueError('GPU output digest changed')
    if (len(fresh)!=3840 or len(reused)!=1530 or done['pretrained_loads']!=2
            or sum(len(r['generated_ids']) for r in fresh)!=done['new_generated_tokens']
            or done['new_generated_tokens']>30720):raise ValueError('generation/load limits changed')
    if any(r['status']!='ok' or not 1<=len(r['generated_ids'])<=8 for r in fresh):raise ValueError('incomplete generation')
    observations=lines(BUNDLE/'evidence/scored_observations.jsonl')
    ids={r['id'] for r in observations}
    if len(ids)!=102 or len({r['source_path'] for r in observations})!=18:raise ValueError('cohort changed')
    indexed=grid([*reused,*(r for r in fresh if r['provenance']=='new')],ids)
    scored=grid(observations,ids)
    panel=set(protocol['schedule']['replay_ids'])
    repeated=[r for r in fresh if r['provenance']=='replay']
    expected={(qid,a,r,s) for qid in panel for a,r in (('native','low'),('native','mid'),('text_only','none')) for s in range(5)}
    if len(panel)!=18 or len(repeated)!=270 or {key(r) for r in repeated}!=expected:raise ValueError('replay coverage changed')
    for row in repeated:replay_equal(row,indexed[key(row)])
    # Reuse the already source-bound content parser; no model/runtime imports.
    parser=runpy.run_path(str(REPO/'experiments/36tar/memory-eval-2b/bundle/experiments/2b-short-event-v1/scoring.py'))['measure']
    gold=None
    if labels is not None:
        if sha(labels)!=manifest['external_labels_sha256']:raise ValueError('external label SHA differs')
        records=read(labels);gold={r['id']:r['correct_index'] for r in records}
        if len(records)!=102 or set(gold)!=ids:raise ValueError('external label cohort differs')
    for k,row in indexed.items():
        displayed=parser(row['text'],5,0)['content_choice']
        choice=(displayed+row['shift'])%5 if displayed is not None else None
        if (row['display_to_original']!=[(j+row['shift'])%5 for j in range(5)]
                or choice!=scored[k]['original_choice'] or row['source_path']!=scored[k]['source_path']):raise ValueError('content mapping changed')
        if gold is not None:
            correct=choice==gold[row['id']] if choice is not None else None
            if correct is not scored[k]['correct']:raise ValueError('official-label rescoring differs')
    summary=read(base/'score/summary.json')
    for arm,res in CONDITIONS:
        selected=[r for r in observations if r['arm']==arm and r['resolution']==res]
        total=summary['totals'][arm+'/'+res]
        correct=sum(r['correct'] is True for r in selected)
        unresolved=sum(r['correct'] is None for r in selected)
        if (len(selected)!=510 or total['content_correct']!=correct or total['unresolved']!=unresolved
                or total['mean_content_accuracy']!=(None if unresolved else correct/510)):raise ValueError('score aggregation differs')
        rows=[r for r in fresh if r['arm']==arm and r['resolution']==res and r['id'] in panel]
        if len(rows)!=90:raise ValueError('matched latency panel changed')
        cost=summary['matched_costs'][arm+'/'+res];times=sorted(r['answer_seconds'] for r in rows)
        if (cost['median_answer_seconds']!=median(times) or cost['p95_answer_seconds']!=times[85]
                or cost['mean_input_tokens']!=sum(r['input_tokens'] for r in rows)/90):raise ValueError('latency/input aggregation differs')
        if arm in ('LS','A06'):
            once=[r for r in rows if r['shift']==0]
            for field,raw in [('mean_encode_seconds','encode_seconds'),('mean_update_seconds','bank_update_seconds')]:
                if cost[field]!=sum(r[raw] for r in once)/18:raise ValueError('encoding/update aggregation differs')
    for name,comparison in summary['paired'].items():
        left,right=name.split('_vs_');la,lr=left.split('/');ra,rr=right.split('/')
        gain=loss=0
        for qid in ids:
            for shift in range(5):
                a=scored[qid,la,lr,shift]['correct'];b=scored[qid,ra,rr,shift]['correct']
                gain+=a is True and b is False;loss+=a is False and b is True
        if gain!=len(comparison['corrections']) or loss!=len(comparison['losses']) or gain-loss!=comparison['net']:raise ValueError('paired gains/losses differ')
    selections=read(base/'cpu/selections.json')
    calls=690+sum(len(row['frames'])//2 for row in selections)
    if calls!=10056 or calls!=done['visual_forward_calls'] or calls!=done['expected_visual_calls']:raise ValueError('visual-forward accounting differs')
    events=lines(base/'gpu/events.jsonl')
    expected={(qid,a,r) for qid in ids for a in ('LS','A06') for r in RESOLUTIONS}
    if len(events)!=612 or {(r['id'],r['arm'],r['resolution']) for r in events}!=expected:raise ValueError('reset coverage differs')
    for row in events:
        state=row['state'];p=RESOLUTIONS[row['resolution']]
        if not row['reset_passed'] or state['tokens']>8*p or len(state['nodes'])>8:raise ValueError('reset/bank bounds failed')
        if state['full_history_features_retained'] or state['language_cache_retained']:raise ValueError('unexpected retained state')
    return dict(status='verified',primary_answers=5100,replayed_answers=270,new_answers=3840,
        tests=72,tiny_answers=9,reset_checks=612,visual_forward_calls=calls,
        execution_source_files=226,official_labels_rescored=gold is not None,
        derived_correctness_flags='statistics checked; official label file required to re-establish correctness' if gold is None else 'recomputed against SHA-pinned labels',
        model_loads_this_verification=0,network_calls=0,writes=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,help='Authorized local SHA-pinned 102-question label file')
    args=parser.parse_args()
    print(json.dumps(verify(args.labels),ensure_ascii=False))
