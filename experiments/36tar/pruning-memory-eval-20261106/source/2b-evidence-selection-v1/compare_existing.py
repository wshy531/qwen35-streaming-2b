"""Exact question/permutation joins across existing methods; no new inference."""
import json,re,sys
from pathlib import Path
import numpy as np
from prepare import ROOT,PREP,read,save,sha

def choice(r):
    m=re.search(r'\b([A-E])\b',r['text'].strip().upper())
    return None if not m else (ord(m.group(1))-65+r['shift'])%5

def main():
    run=ROOT/'runs/2b-evidence-selection-219515'
    tasks=read(PREP/'inputs/requests.json');labels=read(PREP/'audit/labels_LOCAL_ONLY.json')
    mappings=read(ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/evaluation_tasks_LOCAL_ONLY.json')
    local={r.get('original_task_id',r['id']):r['id'] for r in mappings}
    old=read(ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/frozen_branch_outputs_LOCAL_ONLY.json')
    branch={(r['task_id'],r['shift']):r for r in old}
    policies={}
    raw=[json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    for arm in ['recent','full','lexical','time_matched_nodes','nonmatching_nodes']:
        policies[arm]={(r['task_id'],r['shift']):(choice(r)==labels[r['task_id']] if r['available'] else None) for r in raw if r['arm']==arm}
    oldfit=read(ROOT/'runs/2b-read-benefit-219429/fit_result.json')
    for mode in ['recent_video','no_video']:
        routes={r['task_id']:r['route'] for r in oldfit['predictions'][mode]}
        policies['old_gain/'+mode]={(t['id'],s):branch[local[t['id']],s][routes[local[t['id']]]]['correct'] for t in tasks for s in range(5)}
    bundle=read(ROOT/'runs/router_head_training_delivery_20261029_01/heads_all.json')
    sys.path.insert(0,str(ROOT/'scripts'));from router_head_predict import predict
    with np.load(ROOT/'runs/2b-read-benefit-219429/features.npz',allow_pickle=False) as features:
        ids=features['ids'].tolist()
        for name in bundle['heads']:
            mode='no_video' if name.endswith('/no_video') else 'recent_video'
            scores,routes=predict(bundle,name,features[mode]);route=dict(zip(ids,routes.tolist()))
            # Verify old full-102 published aggregate before extracting the subset.
            if name=='round1_balanced/API/recent_video':
                official={r['id'] for r in mappings if r['cohort']=='official102_reference'}
                assert sum(branch[i,s][route[i]]['correct'] for i in official for s in range(5))==130
            policies[name]={(t['id'],s):branch[local[t['id']],s][route[local[t['id']]]]['correct'] for t in tasks for s in range(5)}
    strong=[json.loads(l) for l in (ROOT/'runs/strong-perception-219445/answers.jsonl').read_text(encoding='utf-8').splitlines()]
    strong={(r['task_id'],r['shift'],r['arm']):r for r in strong}
    for arm in ['recent','full']:
        policies['9B/'+arm]={(t['id'],s):choice(strong[local[t['id']],s,arm])==labels[t['id']] for t in tasks for s in range(5)}
    details=read(ROOT/'runs/memory_collab_prepare_20261026_01/partner_readonly/details.json')
    official={t['id'] for t in tasks if t['role']=='official_reference'}
    memory=[]
    for arm in sorted({r['arm'] for r in details}):
        part=[r for r in details if r['arm']==arm and r['id'] in official]
        assert len(part)==230
        memory.append(dict(arm=arm,correct=sum(r['correct'] for r in part),denominator=230))
        if arm=='text_only':
            d={(r['id'],r['shift']):r['correct'] for r in part}
            policies['text_only']={(t['id'],s):d.get((t['id'],s)) for t in tasks for s in range(5)}
    output=[]
    for name,p in policies.items():
        row=dict(method=name)
        for role in ['official_reference','current','old']:
            vals=[p[t['id'],s] for t in tasks if t['role']==role for s in range(5)]
            row[role]=dict(correct=sum(v is True for v in vals),available=sum(v is not None for v in vals))
        cases=sorted({t['case_id'] for t in tasks if t['role']=='old'})
        vals=[(p[c+':late:old',s],p[c+':late:current',s]) for c in cases for s in range(5)]
        row['joint']=dict(correct=sum(a is True and b is True for a,b in vals),available=sum(a is not None and b is not None for a,b in vals))
        output.append(row)
    result=dict(tasks=64,official_questions=46,current_question_times=12,old_question_times=6,
        same_task_methods=output,same_official_46_memory_methods=memory,
        no_new_GPU=True,no_new_model_answers=True,format_not_penalized=True,rotations_not_independent=True,
        time_matched_is_annotation_guided_not_trained_selector=True,
        different_training_data_and_base_model_explicit=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [run/'answers.jsonl',PREP/'inputs/requests.json',
          ROOT/'runs/router_head_training_delivery_20261029_01/heads_all.json',
          ROOT/'runs/strong-perception-219445/answers.jsonl',
          ROOT/'runs/memory_collab_prepare_20261026_01/partner_readonly/details.json']})
    save(run/'cross_method_comparison.json',result)
    for r in output:print(r['method'],[(k,v['correct'],v['available']) for k,v in r.items() if isinstance(v,dict)])
    print('MEMORY 46',memory)
if __name__=='__main__':main()
