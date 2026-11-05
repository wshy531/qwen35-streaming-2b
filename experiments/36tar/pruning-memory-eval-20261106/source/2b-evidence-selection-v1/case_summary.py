"""Question-level and source-level effects; permutations remain correlated."""
import argparse,json,re
from collections import defaultdict
from pathlib import Path
from prepare import PREP,read,save

def main(run,output):
    rows=[json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    tasks=read(PREP/'inputs/requests.json');labels=read(PREP/'audit/labels_LOCAL_ONLY.json')
    byid={t['id']:t for t in tasks};plan={r['task_id']:r for r in read(PREP/'runtime_plan.json')['tasks']}
    correct={};groups=defaultdict(lambda:dict(correct=0,available=0,missing=0))
    for r in rows:
        key=(r['task_id'],r['arm'],r['shift']);t=byid[r['task_id']]
        if r['available']:
            m=re.search(r'\b([A-E])\b',r['text'].strip().upper())
            choice=None if m is None else (ord(m.group(1))-65+r['shift'])%5
            correct[key]=choice==labels[r['task_id']]
        else:correct[key]=None
        group=groups[t['category']+'|'+r['arm']]
        group['missing']+=not r['available']
        group['available']+=r['available'];group['correct']+=correct[key] is True
    question=[]
    for t in tasks:
        counts={}
        for arm in ('recent','full','lexical','time_matched_nodes','nonmatching_nodes'):
            vals=[correct[t['id'],arm,s] for s in range(5)]
            counts[arm]=dict(correct=sum(v is True for v in vals),available=sum(v is not None for v in vals))
        p=plan[t['id']];sel=p['selection']['selected_history_indices']
        question.append(dict(task_id=t['id'],source_name=t['source_name'],role=t['role'],case_id=t.get('case_id'),
            stage=t.get('stage'),counts=counts,selected_visual_tokens=p['selection']['total_visual_tokens'],
            selected_history=[p['nodes'][i] for i in sel],omitted_matching_nodes=p['selection']['omitted_matching_nodes'],
            paired_corrections_vs_full=sum(correct[t['id'],'time_matched_nodes',s] is True and correct[t['id'],'full',s] is False for s in range(5)),
            paired_new_errors_vs_full=sum(correct[t['id'],'time_matched_nodes',s] is False and correct[t['id'],'full',s] is True for s in range(5))))
    sources=[]
    for name in sorted({t['source_name'] for t in tasks}):
        qs=[q for q in question if q['source_name']==name]
        for scope in ('official_reference','development_primary'):
            items=[q for q in qs if (q['role']=='official_reference')==(scope=='official_reference')]
            if items:sources.append(dict(source_name=name,scope=scope,questions=len(items),
                corrections=sum(q['paired_corrections_vs_full'] for q in items),
                new_errors=sum(q['paired_new_errors_vs_full'] for q in items)))
    control_paired=[]
    for role in ('official_reference','current','old'):
        eligible=[(t['id'],s) for t in tasks if t['role']==role for s in range(5)
                  if correct[t['id'],'nonmatching_nodes',s] is not None]
        control_paired.append(dict(role=role,same_question_rotation_denominator=len(eligible),
            scores={a:sum(correct[i,a,s] is True for i,s in eligible)
                for a in ('recent','full','lexical','time_matched_nodes','nonmatching_nodes')}))
    save(output,dict(questions=question,by_source=sources,by_category=dict(groups),
        control_paired_equal_denominator=control_paired,
        old_task_net_gains=sum(q['counts']['time_matched_nodes']['correct']>q['counts']['full']['correct'] for q in question if q['role']=='old'),
        old_task_net_losses=sum(q['counts']['time_matched_nodes']['correct']<q['counts']['full']['correct'] for q in question if q['role']=='old'),
        no_new_answers=True,no_label_changes=True,rotations_not_independent=True))
    print('Question/source breakdown saved; no new model run.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run,a.output)
