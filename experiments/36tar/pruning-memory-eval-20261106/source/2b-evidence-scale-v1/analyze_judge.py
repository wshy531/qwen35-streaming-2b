"""Interpret blinded auxiliary judgments separately from official content scores."""
import argparse,collections,hashlib,json,itertools
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cases=read(a.inputs/'cases_LOCAL_ONLY.json');tasks={x['task_id']:x for x in read(a.inputs/'tasks.json')}
    judged={r['opaque_id']:r['judgment'] for r in read(a.run/'judgments.json')}
    merged=[];tables={};pairs=[];repeats=[]
    for c in cases:
        j=judged.get(c['opaque_id']);issues=[]
        if j is None:issues=['missing_judgment']
        elif j.get('processing_status')!='parsed':issues=['judge_needs_review']
        elif j.get('support')=='sufficient' and j.get('core_error') is True:issues=['sufficient_but_core_contradiction']
        merged.append(dict(**c['LOCAL_ONLY'],opaque_id=c['opaque_id'],task_id=c['task_id'],question=tasks[c['task_id']]['question'],
            proposed_answer=c['proposed_answer'],response=c['response'],judgment=j,review_issues=issues))
    arms=('full','fixed12','heuristic11','mask_ce','text_only');rank={'sufficient':2,'partial':1,'insufficient':0}
    for ds in sorted({r['dataset'] for r in merged}):
        tables[ds]={};sub=[r for r in merged if r['dataset']==ds];idx={(r['question_id'],r['arm']):r for r in sub};qids=sorted({r['question_id'] for r in sub})
        for arm in arms:
            rows=[r for r in sub if r['arm']==arm];usable=[r for r in rows if not r['review_issues']]
            tables[ds][arm]=dict(n=len(rows),official_correct=sum(r['official_correct'] is True for r in rows),
                judged=len(usable),pending=len(rows)-len(usable),support=dict(collections.Counter(r['judgment'].get('support') for r in usable)),
                core_errors=sum(r['judgment'].get('core_error') is True for r in usable),
                core_unverified=sum(r['judgment'].get('core_unverified') is True for r in usable),
                correct_with_sufficient_support=sum(r['official_correct'] is True and r['judgment'].get('support')=='sufficient' for r in usable),
                official_wrong_judge_sufficient=sum(r['official_correct'] is False and r['judgment'].get('support')=='sufficient' for r in usable))
        for aa,bb in itertools.combinations(arms,2):
            win=loss=tie=pending=0;changes=[]
            for qid in qids:
                x,y=idx[qid,aa],idx[qid,bb];jx,jy=x['judgment'],y['judgment']
                if x['review_issues'] or y['review_issues'] or jx.get('support') not in rank or jy.get('support') not in rank:
                    pending+=1;continue
                diff=rank[jx['support']]-rank[jy['support']];win+=diff>0;loss+=diff<0;tie+=diff==0
                if diff:changes.append(dict(question_id=qid,question=x['question'],support_a=jx['support'],support_b=jy['support']))
            pairs.append(dict(dataset=ds,a=aa,b=bb,n=len(qids),a_better=win,b_better=loss,same=tie,uncomparable=pending,changes=changes))
    for r in read(a.inputs/'repeats_LOCAL_ONLY.json'):
        x,y=judged.get(r['repeat_of']),judged.get(r['opaque_id'])
        if x and y:repeats.append(dict(id=r['opaque_id'],support_same=x.get('support')==y.get('support'),core_error_same=x.get('core_error')==y.get('core_error')))
    exact=collections.defaultdict(list)
    for r in merged:exact[r['task_id'],r['response'],r['proposed_answer']].append(r)
    same_text=[]
    for key,rows in exact.items():
        if len(rows)>1 and all(not r['review_issues'] for r in rows):
            same_text.append(dict(question_id=rows[0]['question_id'],arms=[r['arm'] for r in rows],
                support_consistent=len({r['judgment']['support'] for r in rows})==1,
                error_consistent=len({str(r['judgment'].get('core_error')) for r in rows})==1))
    report=dict(tables=tables,paired=pairs,repeats=repeats,identical_response_consistency=same_text,
        not_gold_accuracy=True,subset_is_not_full_300=True,new_training=0)
    (a.run/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.run/'paired_details.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=tables,repeats=repeats,identical_text=same_text),ensure_ascii=False))
if __name__=='__main__':main()
