import argparse,json,collections
from pathlib import Path
from channel_readout import R,ARMS
from experiment import read,save,parse_choice
def analyze(out):
    done=read(out/'completed.json');assert done['status']=='complete' and done['controls']==180 and done['answers']==360
    rows=[json.loads(x) for x in (out/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    qs={q['id']:q for q in read(R/'runs/isolated_recent_prepare_20261103_01/paired_questions.json')}
    labels=read(R/'runs/recent_detail_prepare_20261103_01/labels_LOCAL_ONLY.json')
    index={(r['id'],r['shift'],r['arm']):r for r in rows}
    assert len(rows)==len(index)==360 and set(index)=={(q,s,a) for q in qs for s in range(5) for a in ARMS}
    for (q,s,a),r in index.items():
        l=parse_choice(r['text'],len(qs[q]['options']));selected=None if l is None else (ord(l)-65+s)%len(qs[q]['options'])
        r.update(content_correct=selected==labels[q]['correct_index'],original_selected_index=selected)
    scores={};cases=sorted({q['case_id'] for q in qs.values()})
    for a in ARMS:
        scores[a]={role:dict(correct=sum(index[q,s,a]['content_correct'] for q in qs for s in range(5) if qs[q]['role']==role),answers=sum(5 for q in qs if qs[q]['role']==role)) for role in ('current','old')}
        scores[a]['same_time_both_correct']=sum(all(index[f'{c}:late:{role}',s,a]['content_correct'] for role in ('current','old')) for c in cases for s in range(5))
    per_question={q:{a:sum(index[q,s,a]['content_correct'] for s in range(5)) for a in ARMS} for q in qs}
    comparisons={}
    for a,b in [('joint_KV_recent_GDN','joint'),('isolated_KV_joint_GDN','joint'),('isolated','joint_KV_recent_GDN'),('isolated','isolated_KV_joint_GDN')]:
        comparisons[f'{a}_minus_{b}']={role:dict(gained=sum(index[q,s,a]['content_correct'] and not index[q,s,b]['content_correct'] for q in qs for s in range(5) if qs[q]['role']==role),lost=sum(index[q,s,b]['content_correct'] and not index[q,s,a]['content_correct'] for q in qs for s in range(5) if qs[q]['role']==role)) for role in ('current','old')}
    result=dict(run=done,scores=scores,per_question=per_question,comparisons=comparisons,limits=['18 familiar diagnostic questions from five videos, not 90 independent questions','Bundle interventions may be out of pretraining distribution','Does not prove all causal effects originate in one state channel','No standalone cost comparison'])
    save(out/'analysis.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);analyze(p.parse_args().out)
