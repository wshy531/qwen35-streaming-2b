"""Paired descriptive analysis. Judge estimates are not benchmark gold labels."""
import collections,json
from pathlib import Path
from payloads import R,P,read
O=R/'runs/evidence-demo-judge-20261102-01'
ARMS=('full','fixed12','heuristic11')
def main():
    cases=read(P/'cases_LOCAL_ONLY.json');tasks={t['task_id']:t for t in read(P/'tasks.json')}
    judgments={r['opaque_id']:r['judgment'] for r in read(O/'judgments.json')} if (O/'judgments.json').exists() else {}
    merged=[]
    for c in cases:
        j=judgments.get(c['opaque_id']);m=c['LOCAL_ONLY']
        merged.append(dict(**m,opaque_id=c['opaque_id'],task_id=c['task_id'],question=tasks[c['task_id']]['question'],
            proposed_answer=c['proposed_answer'],response=c['response'],judgment=j))
    tables={};pairs={}
    for dataset in ('NExT-QA','StreamingBench'):
        subset=[r for r in merged if r['dataset']==dataset];table={}
        for arm in ARMS:
            rows=[r for r in subset if r['arm']==arm];jj=[r for r in rows if r['judgment'] is not None and r['judgment'].get('processing_status')=='parsed']
            table[arm]=dict(questions=len(rows),official_correct=sum(r['official_correct'] is True for r in rows),
                official_unresolved=sum(r['official_correct'] is None for r in rows),judged=len(jj),pending=len(rows)-len(jj),
                support=dict(collections.Counter(r['judgment'].get('support') for r in jj)),
                core_contradictions=sum(r['judgment'].get('core_error') is True for r in jj),
                core_unverified=sum(r['judgment'].get('core_unverified') is True for r in jj),
                official_correct_with_sufficient_evidence=sum(r['official_correct'] is True and r['judgment'].get('support')=='sufficient' and r['judgment'].get('core_error') is False for r in jj),
                official_correct_but_core_error=sum(r['official_correct'] is True and r['judgment'].get('core_error') is True for r in jj),
                official_wrong_but_judge_sufficient=sum(r['official_correct'] is False and r['judgment'].get('support')=='sufficient' for r in jj),
                input_tokens=sum(r['input_tokens'] for r in rows),output_tokens=sum(r['output_tokens'] for r in rows))
        tables[dataset]=table
        idx={(r['question_id'],r['arm']):r for r in subset};qids=sorted({r['question_id'] for r in subset})
        changes=[]
        for q in qids:
            b=idx[q,'fixed12'];c=idx[q,'heuristic11'];bj=b['judgment'];cj=c['judgment']
            changes.append(dict(question_id=q,source_id=b['source_id'],question=b['question'],
                official_fixed=b['official_correct'],official_content=c['official_correct'],
                support_fixed=None if bj is None else bj.get('support'),support_content=None if cj is None else cj.get('support'),
                core_error_fixed=None if bj is None else bj.get('core_error'),core_error_content=None if cj is None else cj.get('core_error'),
                same_response=b['response']==c['response']))
        pairs[dataset]=dict(official_gained=sum(c['official_fixed'] is False and c['official_content'] is True for c in changes),
            official_lost=sum(c['official_fixed'] is True and c['official_content'] is False for c in changes),changes=changes)
    repeats=[]
    for rep in read(P/'repeats_LOCAL_ONLY.json'):
        a=judgments.get(rep['repeat_of']);b=judgments.get(rep['opaque_id'])
        if a is not None and b is not None:repeats.append(dict(original=rep['repeat_of'],repeat=rep['opaque_id'],
            support_agrees=a.get('support')==b.get('support'),core_error_agrees=a.get('core_error')==b.get('core_error'),
            original_support=a.get('support'),repeat_support=b.get('support')))
    out=dict(tables=tables,paired=pairs,repeats=repeats,independent_questions=24,not_gold_accuracy=True,
        costs_are_input_output_counts_not_latency=True,new_GPU=0,
        limitations=['opened development sources','12 cases per dataset; descriptive only','judge is fallible',
            'no new evidence removal intervention','recent-eight-pair stream context is not complete history'])
    dest=O if O.exists() else P
    (dest/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (dest/'paired_details.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tables=tables,repeats=repeats),ensure_ascii=False))
if __name__=='__main__':main()
