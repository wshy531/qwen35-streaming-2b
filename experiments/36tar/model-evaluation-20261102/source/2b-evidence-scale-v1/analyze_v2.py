"""Content results and predeclared triggers; never make API calls."""
import argparse,collections,hashlib,json,itertools
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_scale_prepare_20261102_01'
ARMS=('full','fixed12','heuristic11','mask_ce','text_only')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def compare(qs,index,labels,a,b):
    source=collections.defaultdict(list)
    win=loss=tie=0
    for q in qs:
        aa=index[q['id'],a]['original_selected_index']==labels[q['id']]
        bb=index[q['id'],b]['original_selected_index']==labels[q['id']]
        d=int(aa)-int(bb);source[q['source_id']].append(d)
        win+=d>0;loss+=d<0;tie+=d==0
    values=np.array([[sum(v),len(v)] for _,v in sorted(source.items())],dtype=float)
    rng=np.random.default_rng(20261102);draw=rng.integers(0,len(values),size=(4000,len(values)))
    sampled=values[draw].sum(1);ci=np.percentile(sampled[:,0]/sampled[:,1],[2.5,97.5])*100
    diff=100*(win-loss)/len(qs)
    return dict(a=a,b=b,n=len(qs),a_wins=int(win),b_wins=int(loss),same=int(tie),difference_pp=diff,
        source_bootstrap_95pct_pp=ci.tolist(),auxiliary_trigger=bool(abs(diff)<=5 or ci[0]<=0<=ci[1]),
        trigger_is_not_equivalence_or_significance_claim=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=True);panel=read(P/'panel.json');labels=read(P/'labels_LOCAL_ONLY.json');index={};materials={};completed=[]
    for run in a.runs:
        done=read(run/'completed.json');assert done['status']=='complete_generation_only';completed.append(done)
        rows=[json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==750
        for row in rows:
            k=(row['id'],row['arm']);assert k not in index;index[k]=row
        for m in read(run/'materials.json'):assert m['material_id'] not in materials;materials[m['material_id']]=m
    summaries={};comparisons=[];per_type=[]
    for ds in sorted({x['dataset'] for x in completed}):
        qs=[q for q in panel if q['dataset']==ds];summaries[ds]={}
        assert len(qs)==150 and all((q['id'],arm) in index for q in qs for arm in ARMS)
        for arm in ARMS:
            rows=[index[q['id'],arm] for q in qs]
            summaries[ds][arm]=dict(n=150,correct=sum(r['original_selected_index']==labels[r['id']] for r in rows),
                unparsed=sum(r['original_selected_index'] is None for r in rows),
                answer_conflicts=sum(r['answer_conflict'] for r in rows),output_cap_reached=sum(len(r['generated_ids'])==128 for r in rows),
                input_tokens=sum(r['input_tokens'] for r in rows),visual_tokens=sum(r['visual_tokens'] for r in rows),
                note='Unparsed is separate, not evidence of a visual error; original content score retains all 150 questions.')
            for typ in sorted({q['type'] for q in qs}):
                sub=[q for q in qs if q['type']==typ]
                per_type.append(dict(dataset=ds,arm=arm,type=typ,n=len(sub),correct=sum(index[q['id'],arm]['original_selected_index']==labels[q['id']] for q in sub)))
        for aa,bb in itertools.combinations(ARMS,2):comparisons.append(dict(dataset=ds,**compare(qs,index,labels,aa,bb)))
    result=dict(content=summaries,paired=comparisons,per_type=per_type,API=0,training=0,complete_datasets=sorted(summaries),
        expected_datasets=['NExT-QA','StreamingBench'],status='complete' if len(summaries)==2 else 'partial_dataset_report',
        API_automatic_execution=False,subset_denominator_separate=True)
    save(a.out/'analysis.json',result)
    triggered={x['dataset'] for x in comparisons if x['auxiliary_trigger']}
    selected=[q for q in read(P/'auxiliary_panel.json') if q['dataset'] in triggered]
    tasks=[];cases=[]
    for q in selected:
        taskid='task-'+hashlib.sha256(q['id'].encode()).hexdigest()[:16];m=materials[q['id']]
        assert all(f['pts_s']<=q['query_time_s']+1e-6 for f in m['frames'])
        tasks.append(dict(task_id=taskid,question=q['question'],query_time_s=q['query_time_s'],frames=m['frames']))
        for arm in ARMS:
            r=index[q['id'],arm];i=r['original_selected_index']
            cases.append(dict(opaque_id='case-'+hashlib.sha256((q['id']+arm).encode()).hexdigest()[:16],task_id=taskid,
                proposed_answer=None if i is None else q['options'][i],response=r['text'],LOCAL_ONLY=dict(
                    arm=arm,question_id=q['id'],source_id=q['source_id'],dataset=q['dataset'],type=q['type'],
                    official_correct=i==labels[q['id']],input_tokens=r['input_tokens'],output_tokens=len(r['generated_ids']))))
    cases.sort(key=lambda c:c['opaque_id']);repeats=[]
    for ds in sorted(triggered):
        for arm in ARMS:
            c=next(c for c in cases if c['LOCAL_ONLY']['dataset']==ds and c['LOCAL_ONLY']['arm']==arm)
            repeats.append(dict(c,opaque_id='repeat-'+hashlib.sha256(c['opaque_id'].encode()).hexdigest()[:16],repeat_of=c['opaque_id']))
    save(a.out/'tasks.json',tasks);save(a.out/'cases_LOCAL_ONLY.json',cases);save(a.out/'repeats_LOCAL_ONLY.json',repeats)
    save(a.out/'auxiliary_manifest.json',dict(triggered_datasets=sorted(triggered),tasks=len(tasks),cases=len(cases),repeats=len(repeats),
        calibration=20,maximum_API_requests=20+len(tasks)+len(cases)+len(repeats),selection_precedes_predictions=True,paid_calls=0))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
