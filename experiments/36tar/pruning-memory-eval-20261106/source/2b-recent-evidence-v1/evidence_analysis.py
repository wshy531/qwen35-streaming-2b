"""Score selection of frozen answers; routing usefulness is not sufficiency truth."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import sys
from gate_common import ROOT, INPUTS, read, save

def evaluate(decisions):
    tasks=read(INPUTS/'evaluation_tasks_LOCAL_ONLY.json')
    rows=read(INPUTS/'frozen_branch_outputs_LOCAL_ONLY.json')
    expected={(t['id'],s) for t in tasks for s in range(5)}
    if set(decisions)!=expected or any(v not in ('recent','full') for v in decisions.values()):
        raise ValueError('all 600 decisions required')
    if len(rows)!=600 or {(r['task_id'],r['shift']) for r in rows}!=expected:
        raise ValueError('missing or duplicate frozen answer')
    byid={t['id']:t for t in tasks}; groups=defaultdict(Counter);sources=defaultdict(Counter);outputs={}
    for row in rows:
        key=(row['task_id'],row['shift']);t=byid[key[0]];route=decisions[key]
        selected=row[route];a,b=row['recent'],row['full']
        c=groups[(t['cohort'],t['role'])]
        c['denominator']+=1;c['correct']+=int(selected['correct'])
        c['chose_recent']+=int(route=='recent');c['chose_full']+=int(route=='full')
        c['recent_only']+=int(a['correct'] and not b['correct']);c['full_only']+=int(b['correct'] and not a['correct'])
        c['both']+=int(a['correct'] and b['correct']);c['neither']+=int(not a['correct'] and not b['correct'])
        c['missed_existing_full_only']+=int(route=='recent' and b['correct'] and not a['correct'])
        c['harmed_existing_recent_only']+=int(route=='full' and a['correct'] and not b['correct'])
        s=sources[(t['cohort'],t['source_name'],t['role'])];s['denominator']+=1;s['correct']+=int(selected['correct'])
        outputs[key]=selected['correct']
    cases=defaultdict(dict)
    for t in tasks:
        if t.get('case_id'):cases[t['case_id']][(t.get('stage'),t['role'])]=t['id']
    joint=[dict(case_id=case,both_old_and_current_correct=sum(
        outputs[(q[('late','current')],s)] and outputs[(q[('late','old')],s)] for s in range(5)),denominator=5)
        for case,q in sorted(cases.items())]
    return dict(groups=[dict(cohort=c,role=r,**v) for (c,r),v in sorted(groups.items())],
        by_source=[dict(cohort=c,source=s,role=r,**v) for (c,s,r),v in sorted(sources.items())],
        joint_old_current=joint,question_rotation_rows=600,all_outputs_reused=True,new_QA_answers=0,
        rotations_are_not_independent_samples=True,sufficiency_ground_truth_available=False)

def baselines():
    tasks=read(INPUTS/'evaluation_tasks_LOCAL_ONLY.json')
    lexical=read(INPUTS/'lexical_decisions.json')
    return {name:evaluate({(t['id'],s):(lexical[t['router_id']]['route'] if name=='lexical' else name)
        for t in tasks for s in range(5)}) for name in ('recent','full','lexical')}

def decision(candidate, text_control, base):
    sys.path.insert(0,str(ROOT/'experiments/2b-temporal-router-v1'))
    from route_analysis import development_decision
    result=development_decision(candidate,base)
    total=lambda x:sum(g['correct'] for g in x['groups'])
    group=lambda x,role:next(g['correct'] for g in x['groups'] if g['cohort']=='transition6_development' and g['role']==role)
    result['gates']['beats_same_prompt_without_video_total']=total(candidate)>total(text_control)
    result['gates']['current_not_below_no_video']=group(candidate,'current')>=group(text_control,'current')
    result['gates']['old_not_below_no_video']=group(candidate,'old')>=group(text_control,'old')
    result['passed']=all(result['gates'].values())
    result['action']='prepare_source_isolated_validation' if result['passed'] else 'stop_no_prompt_or_threshold_sweep'
    return result

def analyze(run,output):
    rows=[__import__('json').loads(l) for l in (run/'gate_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={(r['id'],s,m) for r in read(INPUTS/'requests.json') for s in range(5) for m in ('recent_video','no_video')}
    keyed={(r['task_id'],r['shift'],r['mode']):r for r in rows}
    if len(rows)!=len(expected) or set(keyed)!=expected:raise ValueError('complete unique 1200 routes required')
    policies={m:evaluate({(t,s):keyed[(t,s,m)]['decision']['route'] for t,s,mode in expected if mode==m})
        for m in ('recent_video','no_video')}
    base=baselines()
    result=dict(status='complete_fixed_answer_selection',policies=policies,baselines=base,
        development_decision=decision(policies['recent_video'],policies['no_video'],base),
        component_seconds={m:sum(r['seconds'] for r in rows if r['mode']==m) for m in policies},
        route_changes_when_removing_video=sum(keyed[(t,s,'recent_video')]['decision']['route']!=
            keyed[(t,s,'no_video')]['decision']['route'] for t,s,mode in expected if mode=='recent_video'),
        no_new_QA=True,not_end_to_end=True,default_enable=False,formal_generalization_claim=False)
    save(output,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.run:analyze(a.run,a.output)
    else:save(a.output,baselines())
