"""Reuse the frozen old lexical policy on scored, identical-prompt branch answers."""
import argparse,json,hashlib,sys,collections
from pathlib import Path
R=Path(__file__).resolve().parents[2]
POLICY=R/'experiments/2b-temporal-router-v1/temporal_policy.py'
sys.path.insert(0,str(POLICY.parent))
from temporal_policy import lexical_route,VERSION

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main(out,panel):
    assert read(out/'completed.json')['status']=='complete'
    rows=[json.loads(x) for x in (out/'scored_LOCAL_ONLY.jsonl').read_text(encoding='utf-8').splitlines()]
    if panel=='sember':
        prep=R/'runs/isolated_recent_prepare_20261103_01'
        qs={('official',q['question_id']):q for q in read(prep/'requests.json')}
        qs.update({('paired',q['id']):q for q in read(prep/'paired_questions.json')})
        keyed={(r['panel'],r['id'],r['shift'],r['arm']):r for r in rows}
        keys=sorted({k[:3] for k in keyed})
    else:
        prep=R/'runs/protection_streamingbench_prepare_20261103_01'
        qs={('stream',q['id']):q for q in read(prep/'questions.json')}
        keyed={('stream',r['id'],0,r['arm']):r for r in rows};keys=sorted({k[:3] for k in keyed})
    selections=[]
    for p,q,s in keys:
        question=qs[p,q];decision=lexical_route(question['question'])
        arm='joint' if decision['route']=='full' else 'recent'
        chosen=keyed[p,q,s,arm];iso=keyed[p,q,s,'isolated']
        field='correct' if panel=='sember' else 'content_correct'
        selections.append(dict(panel=p,id=q,shift=s,role=question.get('role'),type=question.get('question_category',question.get('type')),
            selected_arm=arm,reason=decision,correct=chosen[field],isolated_correct=iso[field]))
    def summarize(rs):
        return dict(answers=len(rs),lexical_correct=sum(x['correct'] for x in rs),isolated_correct=sum(x['isolated_correct'] for x in rs),
            isolated_gained=sum(x['isolated_correct'] and not x['correct'] for x in rs),
            isolated_lost=sum(x['correct'] and not x['isolated_correct'] for x in rs),
            selected_arms=dict(collections.Counter(x['selected_arm'] for x in rs)))
    groups={p:summarize([x for x in selections if x['panel']==p]) for p in sorted({x['panel'] for x in selections})}
    if panel=='sember':
        groups['paired_by_role']={role:summarize([x for x in selections if x['panel']=='paired' and x['role']==role]) for role in ('current','old')}
        chosen={(x['id'],x['shift']):x for x in selections if x['panel']=='paired'}
        cases=sorted({q['case_id'] for (p,_),q in qs.items() if p=='paired'})
        groups['paired_same_time_both_correct']=sum(all(chosen[f'{c}:late:{r}',s]['correct'] for r in ('current','old')) for c in cases for s in range(5))
    result=dict(policy_version=VERSION,policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest(),
        provenance='CPU selection among already generated identical-prompt answers; no new model calls; unchanged historical lexical policy',
        scoring='Selected-answer content; no explanation judging',groups=groups,selections=selections)
    with (out/'lexical_replay.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(groups,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('panel',choices=('sember','stream'));a=p.parse_args();main(a.out,a.panel)
