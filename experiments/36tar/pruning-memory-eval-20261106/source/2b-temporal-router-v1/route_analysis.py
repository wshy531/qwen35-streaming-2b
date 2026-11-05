"""Offline selection of unchanged cached answers; analytical oracles never infer."""
from collections import Counter,defaultdict
import argparse
from pathlib import Path
from route_common import INPUTS,read,save


def evaluate(decisions,tasks,branches):
    expected={t['router_id'] for t in tasks}
    if set(decisions)!=expected or any(v not in ('recent','full') for v in decisions.values()):
        raise ValueError('exactly one valid route per question required')
    bytask={t['id']:t for t in tasks}
    if len(bytask)!=len(tasks):raise ValueError('duplicate evaluation task')
    keys={(r['task_id'],r['shift']) for r in branches}
    if len(keys)!=len(branches) or keys!={(t['id'],s) for t in tasks for s in range(5)}:
        raise ValueError('missing/duplicate cached answer pair')
    groups=defaultdict(Counter);sources=defaultdict(Counter);outputs={}
    keyed={(r['task_id'],r['shift']):r for r in branches}
    for r in branches:
        task=bytask[r['task_id']];route=decisions[task['router_id']]
        selected=r[route];a,b=r['recent'],r['full']
        g=groups[(task['cohort'],task['role'])]
        g['denominator']+=1;g['correct']+=int(selected['correct'])
        g['chose_recent']+=int(route=='recent');g['chose_full']+=int(route=='full')
        g['recent_only']+=int(a['correct'] and not b['correct'])
        g['full_only']+=int(b['correct'] and not a['correct'])
        g['both']+=int(a['correct'] and b['correct'])
        g['neither']+=int(not a['correct'] and not b['correct'])
        g['added_errors_vs_full']+=int(b['correct'] and not selected['correct'])
        g['corrections_vs_full']+=int(selected['correct'] and not b['correct'])
        s=sources[(task['cohort'],task['source_name'],task['role'])]
        s['denominator']+=1;s['correct']+=int(selected['correct'])
        outputs[(task['id'],r['shift'])]=selected['correct']
    paired=defaultdict(dict)
    for task in tasks:
        if task.get('case_id'):paired[task['case_id']][(task.get('stage'),task['role'])]=task['id']
    joint=[]
    for case,q in sorted(paired.items()):
        late,old=q[('late','current')],q[('late','old')]
        joint.append(dict(case_id=case,both_old_and_current_correct=sum(outputs[(late,s)] and outputs[(old,s)] for s in range(5)),denominator=5))
    return dict(groups=[dict(cohort=c,role=r,**v) for (c,r),v in sorted(groups.items())],
        by_source=[dict(cohort=c,source=s,role=r,**v) for (c,s,r),v in sorted(sources.items())],
        joint_old_current=joint,question_rotation_rows=len(branches),
        unique_questions=len(expected),all_outputs_reused=True,new_QA_answers=0,
        rotations_are_not_independent_samples=True,official_reference_not_clean_current_test=True)


def baseline_audit():
    tasks=read(INPUTS/'evaluation_tasks_LOCAL_ONLY.json')
    rows=read(INPUTS/'frozen_branch_outputs_LOCAL_ONLY.json')
    requests=read(INPUTS/'router_requests.json')
    lexical={k:v['route'] for k,v in read(INPUTS/'lexical_decisions.json').items()}
    totals=defaultdict(Counter);bytask={t['id']:t for t in tasks}
    for row in rows:
        key=bytask[row['task_id']]['router_id']
        for branch in ('recent','full'):totals[key][branch]+=int(row[branch]['correct'])
    consistent={k:('recent' if v['recent']>v['full'] else 'full') for k,v in totals.items()}
    headroom=[]
    for key,v in totals.items():
        gain=max(v['recent'],v['full'])-v[lexical[key]]
        if gain:
            headroom.append(dict(router_id=key,possible_additional_correct=gain,
                source_names=sorted({t['source_name'] for t in tasks if t['router_id']==key})))
    from temporal_policy import HISTORY_PATTERN
    import re
    flags=Counter()
    for r in requests:
        q=r['question']
        if re.search(r'\b(now|currently|here|from where)\b',q,re.I) and re.search(HISTORY_PATTERN,q,re.I):
            flags['present_surface_with_history_or_relation_marker']+=1
        if re.search(r'\b(how many|so far)\b',q,re.I):flags['count_or_cumulative_wording']+=1
        if re.search(r'\b(compare\w*|in what order|in what sequence|before|after)\b',q,re.I):flags['relation_or_order_wording']+=1
    return dict(status='offline_pre_model_headroom_audit',
        policies={name:evaluate(d,tasks,rows) for name,d in (
            ('recent',{k:'recent' for k in totals}),('full',{k:'full' for k in totals}),
            ('lexical',lexical),('question_constant_oracle_DIAGNOSTIC',consistent))},
        lexical_headroom=headroom,headroom_questions=len(headroom),
        headroom_sources=len({s for h in headroom for s in h['source_names']}),
        question_text_pattern_counts=dict(flags),
        oracle_uses_outcomes_only_for_analysis=True,oracle_not_exported_as_runtime_input=True,
        labels_not_accessed_by_router=True,new_model_forwards=0,GPU_jobs=0)


def analyze_run(run,output):
    predictions=read(run/'router_predictions.json')
    requests=read(INPUTS/'router_requests.json')
    expected={r['id'] for r in requests}
    if len(predictions)!=len(expected) or {r['id'] for r in predictions}!=expected:
        raise ValueError('complete unique router predictions required')
    decisions={r['id']:r['result']['route'] for r in predictions}
    result=evaluate(decisions,read(INPUTS/'evaluation_tasks_LOCAL_ONLY.json'),read(INPUTS/'frozen_branch_outputs_LOCAL_ONLY.json'))
    baseline=baseline_audit()
    result['baselines']=baseline['policies']
    result['development_decision']=development_decision(result,baseline['policies'])
    result['model_router_only_not_new_QA']=True
    result['router_total_seconds']=sum(r['total_router_seconds'] for r in predictions)
    result['router_mean_seconds_per_unique_question']=result['router_total_seconds']/len(predictions)
    result['not_end_to_end_video_timing']=True
    save(output,result)


def development_decision(candidate,policies):
    def count(value,cohort,role):
        return next(g['correct'] for g in value['groups'] if g['cohort']==cohort and g['role']==role)
    def source_counts(value):
        c=defaultdict(int)
        for row in value['by_source']:c[row['source']]+=row['correct']
        return c
    lexical=policies['lexical'];full=policies['full']
    old_key=('official102_reference','unpartitioned_official_reference')
    dev='transition6_development'
    source_candidate=source_counts(candidate);source_rule=source_counts(lexical)
    improved={s for s,v in source_candidate.items() if v>source_rule[s]}
    joint=lambda p:sum(r['both_old_and_current_correct'] for r in p['joint_old_current'])
    gates=dict(
        official_reference_beats_rule=count(candidate,*old_key)>count(lexical,*old_key),
        official_reference_not_below_full=count(candidate,*old_key)>=count(full,*old_key),
        reviewed_current_not_below_rule=count(candidate,dev,'current')>=count(lexical,dev,'current'),
        reviewed_current_above_full=count(candidate,dev,'current')>count(full,dev,'current'),
        reviewed_old_not_below_full=count(candidate,dev,'old')>=count(full,dev,'old'),
        joint_old_current_not_below_rule=joint(candidate)>=joint(lexical),
        at_least_two_sources_better_than_rule=len(improved)>=2)
    return dict(gates=gates,passed=all(gates.values()),sources_better_than_rule=len(improved),
        action='prepare_independent_validation' if all(gates.values()) else 'stop_candidate_no_prompt_sweep',
        default_enable=False,formal_capability_claim=False,
        cost_scope='router component measured; independent end-to-end cost still unmeasured')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.run:analyze_run(a.run,a.output)
    else:save(a.output,baseline_audit())
