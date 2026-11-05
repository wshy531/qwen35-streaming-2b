"""CPU-only audit of frozen routes. No fitting, new model answers or pooled score."""
from pathlib import Path
import sys, json, re, hashlib, collections
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/memory_route_audit_20261103_01'
FILES = {}

def read(name):
    p = ROOT / name
    FILES[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8'))

def rows(name):
    p = ROOT / name
    FILES[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

def index(data):
    result = {(r['task_id'], r['shift'], r['arm']): r for r in data}
    assert len(result) == len(data), 'duplicate answers'
    return result

def parsed(text, shift, n):
    m = re.search(r'\b([A-E])\b', text.strip().upper())
    return (ord(m[1])-65+shift) % n if m and ord(m[1])-65 < n else None

def summarize(tasks, k):
    out = dict(questions=len(tasks), sources=len({x['source'] for x in tasks}),
               answers=len(tasks)*k, permutations_per_question=k)
    for method in ('recent', 'full', 'lexical', 'api33'):
        out[method] = sum(sum(x['correct'][method]) for x in tasks)
    out['both_wrong'] = sum(sum(not a and not b for a,b in zip(x['correct']['recent'],x['correct']['full'])) for x in tasks)
    out['oracle_per_permutation'] = out['answers'] - out['both_wrong']
    out['oracle_one_route_per_question'] = sum(max(sum(x['correct']['recent']),sum(x['correct']['full'])) for x in tasks)
    for method in ('api33', 'lexical'):
        value = collections.Counter()
        for x in tasks:
            assert len(set(x['routes'][method])) == 1, (x['id'], 'route varies with options')
            value['questions_reading_history'] += x['routes'][method][0] == 'full'
            rc, fc = sum(x['correct']['recent']),sum(x['correct']['full'])
            value['questions_with_equal_branch_scores'] += rc == fc
            value['questions_chose_lower_scoring_branch'] += sum(x['correct'][method]) < max(rc,fc)
            for r,f,route in zip(x['correct']['recent'],x['correct']['full'],x['routes'][method]):
                value['harmful_history_calls'] += route == 'full' and r and not f
                value['useful_history_calls'] += route == 'full' and f and not r
                value['missed_history_benefits'] += route == 'recent' and f and not r
                value['avoided_history_harm'] += route == 'recent' and r and not f
        value['recoverable_gap_unified_route'] = out['oracle_one_route_per_question'] - out[method]
        out[method+'_route_audit'] = dict(value)
    return out

def original120():
    base='runs/recent_evidence_prepare_20261028_01/inputs/'
    tasks=read(base+'evaluation_tasks_LOCAL_ONLY.json')
    req={r['id']:r for r in read(base+'requests.json')}
    branches={(r['task_id'],r['shift']):r for r in read(base+'frozen_branch_outputs_LOCAL_ONLY.json')}
    bundle=read('runs/router_head_training_delivery_20261029_01/heads_all.json')
    fp=ROOT/'runs/2b-read-benefit-219429/features.npz'
    FILES[str(fp.relative_to(ROOT))]=hashlib.sha256(fp.read_bytes()).hexdigest()
    sys.path.insert(0,str(ROOT/'scripts'))
    from router_head_predict import predict
    sys.path.insert(0,str(ROOT/'experiments/2b-temporal-router-v1'))
    from temporal_policy import lexical_route
    with np.load(fp,allow_pickle=False) as f:
        scores,routes=predict(bundle,'round1_balanced/API/recent_video',f['recent_video'])
        route=dict(zip(f['ids'].tolist(),routes.tolist()))
    result=[]
    for t in tasks:
        tid=t['id']; q=req[tid]; r={'api33':route[tid],'lexical':lexical_route(q['question'])['route']}
        correct={m:[bool(branches[tid,s][m]['correct']) for s in range(5)] for m in ('recent','full')}
        correct.update({m:correct[v].copy() for m,v in r.items()})
        result.append(dict(id=tid,original_id=t.get('original_task_id',tid),source=t['source_name'],
            question=q['question'],group=t['cohort']+'/'+t['role'],correct=correct,
            routes={m:[v]*5 for m,v in r.items()}))
    assert summarize(result,5)['api33']==192
    return result, 5

def followup100():
    tasks=read('runs/api33_followup_prepare_20261031_01/requests.json')
    labels=read('runs/api33_followup_prepare_20261031_01/labels_LOCAL_ONLY.json')
    old=index(rows('runs/selector-qa-219534/answers.jsonl'))
    new=index(rows('runs/api33-followup-219752/answers.jsonl'))
    # Reusing the recent arm is valid only if the two shared arms still match exactly.
    for (tid,s,arm),v in new.items():
        if arm in ('full','lexical'):
            for field in ('generated_ids','input_ids_sha256','position_ids_sha256'):
                assert v[field]==old[tid,s,arm][field], (tid,s,arm,field)
    result=[]
    for q in tasks:
        tid=q['id']; correct={}; routes={}
        for name,arm,data in [('recent','recent',old),('full','full',new),('lexical','lexical',new),('api33','API33_balanced_visual',new)]:
            rr=[data[tid,s,arm] for s in range(5)]
            correct[name]=[parsed(v['text'],s,5)==labels[tid] for s,v in enumerate(rr)]
            if name in ('lexical','api33'):
                routes[name]=[v['selector']['reading'] for v in rr]
                for s,v in enumerate(rr):
                    target=old[tid,s,routes[name][s]]
                    for field in ('generated_ids','input_ids_sha256','position_ids_sha256'):
                        assert v[field]==target[field], (tid,s,name,field)
        result.append(dict(id=tid,original_id=tid,source=q['source_name'],question=q['question'],
            group=q['evaluation_group']+'/'+q['role'],correct=correct,routes=routes))
    assert summarize(result,5)['api33']==172
    return result,5

def streaming40():
    base='runs/streamingbench_router_prepare_20261101_01/'
    tasks=read(base+'profile_requests.json'); data=index(rows(base+'GPU_results_219953/answers.jsonl'))
    labels=read('runs/streamingbench_download_prepare_20261030_01/eval_inputs/labels_LOCAL_ONLY.json')
    result=[]
    for q in tasks:
        tid=q['id']; correct={}; routes={}
        for name,arm in [('recent','recent'),('full','LS_joint'),('lexical','temporal_word_rule'),('api33','API33_balanced_visual')]:
            rr=[data[tid,s,arm] for s in range(4)]
            correct[name]=[parsed(v['text'],s,4)==ord(labels[tid]['answer'])-65 for s,v in enumerate(rr)]
            if name in ('lexical','api33'):
                routes[name]=[v['decision']['route'] for v in rr]
                for s,v in enumerate(rr):
                    target=data[tid,s,'LS_joint' if routes[name][s]=='full' else 'recent']
                    for field in ('generated_ids','input_ids_sha256','position_ids_sha256'):
                        assert v[field]==target[field], (tid,s,name,field)
        result.append(dict(id=tid,original_id=tid,source=q['media_name'],question=q['question'],group=q['task_type'],correct=correct,routes=routes))
    assert summarize(result,4)['api33']==89
    return result,4

def main():
    assert not OUT.exists(), 'do not overwrite evidence'
    datasets={}; allrows=[]
    for name,load in [('original120',original120),('followup100',followup100),('streaming40',streaming40)]:
        tasks,k=load(); stat=summarize(tasks,k)
        stat['by_group']={g:summarize([x for x in tasks if x['group']==g],k) for g in sorted({x['group'] for x in tasks})}
        stat['by_source']={g:summarize([x for x in tasks if x['source']==g],k) for g in sorted({x['source'] for x in tasks})}
        datasets[name]=stat
        for x in tasks:
            x.update(dataset=name,permutations=k,api33_gap=max(sum(x['correct']['recent']),sum(x['correct']['full']))-sum(x['correct']['api33']))
            allrows.append(x)
    overlaps={}
    for a,b in [('original120','followup100'),('original120','streaming40'),('followup100','streaming40')]:
        aa=[x for x in allrows if x['dataset']==a]; bb=[x for x in allrows if x['dataset']==b]
        overlaps[a+'__'+b]=dict(task_ids=len({x['original_id'] for x in aa}&{x['original_id'] for x in bb}),
            source_names=len({x['source'] for x in aa}&{x['source'] for x in bb}))
    for p in [Path(__file__), ROOT/'scripts/router_head_predict.py',ROOT/'experiments/2b-temporal-router-v1/temporal_policy.py']:
        FILES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(datasets=datasets,overlap_by_identity_only=overlaps,sources_sha256=FILES,no_training=True,no_new_answers=True,
        oracle_is_retrospective_not_automatic_score=True,do_not_pool_datasets=True,
        limitation='Original120 only frozen correctness; newer two panels additionally verify selected answer/input/position endpoint identity. No semantic source dedup or new evidence audit.')
    OUT.mkdir(parents=True)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'cases_LOCAL_ONLY.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allrows),encoding='utf-8')
    for name,v in datasets.items():
        print(name,json.dumps({k:v[k] for k in ['questions','sources','answers','recent','full','lexical','api33','both_wrong','oracle_one_route_per_question','api33_route_audit']},ensure_ascii=False))

if __name__=='__main__':main()
