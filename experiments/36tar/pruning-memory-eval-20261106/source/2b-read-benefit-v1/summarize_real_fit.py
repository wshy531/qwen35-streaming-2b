"""Report fixed OOF policies, paired source uncertainty and extra feature cost."""
from pathlib import Path
from collections import defaultdict
import argparse,json,random,statistics
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main(run):
    fit=read(run/'fit_result.json');labels=read(ROOT/'runs/read_benefit_prepare_20261028_01/audit/policy_targets_LOCAL_ONLY.json')
    rows=read(ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/frozen_branch_outputs_LOCAL_ONLY.json')
    tasks={r['id']:r for r in read(ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/evaluation_tasks_LOCAL_ONLY.json')}
    lexical=read(ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/lexical_decisions.json')
    preds={mode:{r['task_id']:r for r in records} for mode,records in fit['predictions'].items()}
    group={r['task_id']:r['leakage_group'] for r in labels};fold={r['task_id']:r['fold'] for r in labels}
    def route(name,task):
        if name in ('recent','full'):return name
        if name=='lexical':return lexical[tasks[task]['router_id']]['route']
        return preds[name][task]['route']
    comparisons={}
    for cohort in ('official102_reference','transition6_development'):
        selected=[r for r in rows if tasks[r['task_id']]['cohort']==cohort]
        by_group=defaultdict(list)
        for row in selected:by_group[group[row['task_id']]].append(row)
        for base in ('recent','full','lexical','no_video'):
            values=[];wins=losses=0
            for row in selected:
                q=row['task_id'];a=row[route('recent_video',q)]['correct'];b=row[route(base,q)]['correct'];wins+=a and not b;losses+=b and not a
            rng=random.Random(20261028);keys=sorted(by_group)
            for _ in range(2000):
                sampled=[r for k in rng.choices(keys,k=len(keys)) for r in by_group[k]]
                values.append(sum(int(r[route('recent_video',r['task_id'])]['correct'])-int(r[route(base,r['task_id'])]['correct']) for r in sampled)/len(sampled))
            values.sort();comparisons[cohort+'/'+base]=dict(paired_corrected=wins,paired_regressed=losses,rotation_rows=len(selected),source_connected_groups=len(keys),
                bootstrap_CI95_delta_accuracy=[values[49],values[1949]],descriptive_only=True,rotations_not_independent=True)
    per_task=[]
    for label in labels:
        q=label['task_id'];entry=dict(task_id=q,fold=fold[q],group=group[q],target_gain=label['target_gain'],fit_weight=label['fit_weight'],
            cohort=label['cohort'],role=label['role'],source_name=label['source_name'])
        for mode in ('recent_video','no_video'):entry[mode]=preds[mode][q]
        per_task.append(entry)
    timings=[json.loads(s) for s in (run/'timings.jsonl').read_text().splitlines()]
    costs={str(e):dict(n=sum(r['evidence']==e for r in timings),total_seconds=sum(r['seconds'] for r in timings if r['evidence']==e),
        median_seconds=statistics.median(r['seconds'] for r in timings if r['evidence']==e)) for e in (True,False)}
    out=dict(status='real_grouped_OOF_fit_analyzed',paired_comparisons=comparisons,per_task=per_task,feature_cost=costs,
        full_routes_by_mode={m:sum(r['route']=='full' for r in p.values()) for m,p in preds.items()},
        fold_policy_counts={m:{str(f):dict(tasks=sum(r['fold']==f for r in p.values()),full=sum(r['fold']==f and r['route']=='full' for r in p.values())) for f in range(3)} for m,p in preds.items()},
        development_decision=fit['development_decision'],no_new_QA=True,not_end_to_end_cost=True,no_threshold_or_seed_search=True)
    (run/'fit_analysis.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='per_task'},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
