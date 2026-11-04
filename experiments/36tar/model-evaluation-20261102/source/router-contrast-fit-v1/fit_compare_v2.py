"""One fixed33 repair fit, old models and no-image controls, no tuning loop."""
from pathlib import Path
from collections import Counter
import argparse,json,hashlib,sys,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/router-bank-head-fit-v1'))
from fixed_fit import fit,bundle,BenefitHead,agreement
sys.path.insert(0,str(ROOT/'experiments/router-contrast-v1'))
from time_constraint import constrain

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def restore(v):
    h=BenefitHead()
    for k in ('coefficient','mean','scale'):setattr(h,k,np.asarray(v[k],float))
    return h

def main(feature_run,out):
    if out.exists() or not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('new owned fit output')
    prep=ROOT/'runs/router_contrast_features_prepare_20261101_01'
    plan=read(prep/'fixed_fit_plan.json');labels=read(prep/'labels_LOCAL_ONLY.json');by={r['id']:r for r in labels}
    if read(feature_run/'completed.json')['status']!='complete':raise ValueError('incomplete real features')
    inputs=[prep/'fixed_fit_plan.json',prep/'labels_LOCAL_ONLY.json',feature_run/'features.npz',
        ROOT/'runs/router-bank-features-219930/features.npz',ROOT/'runs/router-bank-fit-20261101-01/models.json',
        ROOT/'runs/2b-read-benefit-219429/features.npz',ROOT/'experiments/router-contrast-v1/time_constraint.py',
        Path(__file__),ROOT/'experiments/router-bank-head-fit-v1/fixed_fit.py']
    inputs += [ROOT/'runs/router-bank-teacher-20261101-01/feature_requests.json',prep/'requests.json']
    request_by={t['id']:t for t in read(inputs[-2])+read(inputs[-1])}
    if set(request_by)!=set(by):raise ValueError('question/label ID contract')
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    features={}
    for p in (inputs[3],inputs[2]):
        with np.load(p,allow_pickle=False) as f:
            for i,s in enumerate(f['ids'].tolist()):
                if s in features:raise ValueError('feature IDs overlap')
                features[s]={m:f[m][i].copy() for m in ('recent_video','no_video')}
    if set(features)!=set(by):raise ValueError('combined label/features IDs mismatch')
    selected=plan['training_ids'];assert len(selected)==195 and all(by[s]['role']=='train' for s in selected)
    prior=read(inputs[4]);models={name:prior[name] for name in ['old_frozen195/recent_video','old_frozen195/no_video','new_matched_N/recent_video','new_matched_N/no_video']}
    for mode in ('recent_video','no_video'):
        h=fit(np.stack([features[s][mode] for s in selected]),[by[s]['signed_route_target'] for s in selected],[by[s]['source_group'] for s in selected])
        models['repaired195/'+mode]=bundle(h,selected)
    ids=list(by);metrics={};predictions={};contrasts={}
    families=read(ROOT/'runs/router_contrast_prepare_20261101_01/families_LOCAL_ONLY.json')
    for name,v in models.items():
        mode=name.split('/')[-1];h=restore(v);scores=h.predict_gain(np.stack([features[s][mode] for s in ids]));pred={s:1 if scores[i]>0 else -1 for i,s in enumerate(ids)}
        predictions[name]=[dict(id=s,score=float(scores[i]),signed_route=pred[s]) for i,s in enumerate(ids)]
        metrics[name]={}
        for role in ('train','dev','source_confirmation'):
            metrics[name][role]={}
            for group in ('original','new_views','all'):
                ss=[s for s in ids if by[s]['role']==role and (group=='all' or ('kind' in by[s])==(group=='new_views'))]
                metrics[name][role][group]=agreement([by[s]['signed_route_target'] for s in ss],[pred[s] for s in ss],[by[s]['source_group'] for s in ss])
        contrasts[name]={}
        for role in ('train','dev','source_confirmation'):
            ff=[f for f in families if f['role']==role and f['early_id'] in by]
            flip=[f for f in ff if by[f['early_id']]['signed_route_target']==-1]
            late=[r for r in labels if r['role']==role and r.get('kind')=='later' and r['signed_route_target']==-1]
            contrasts[name][role]=dict(accepted_early_families=len(ff),true_flip_families=len(flip),
                both_opposite_routes_correct=sum(pred[f['early_id']]==-1 and pred[f['original_id']]==1 for f in flip),
                later_recent_controls=len(late),later_recent_correct=sum(pred[r['id']]==-1 for r in late))
    # Existing answers remain fixed and correlated by source and option permutation.
    sys.path.insert(0,str(ROOT/'experiments/2b-recent-evidence-v1'))
    from evidence_analysis import evaluate,baselines
    request=read(ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/requests.json');task_by={t['id']:t for t in request}
    regressions={}
    with np.load(inputs[5],allow_pickle=False) as f:
        rrids=f['ids'].tolist()
        for name,v in models.items():
            mode=name.split('/')[-1];scores=restore(v).predict_gain(f[mode]);decisions={};guarded={}
            for i,s in enumerate(rrids):
                route='full' if scores[i]>0 else 'recent';t=task_by[s]
                q=constrain(t['question'],[p['pts_s'] for p in t['recent_frames']],t['query_time_s'],route)
                for shift in range(5):decisions[s,shift]=route;guarded[s,shift]=q['route'] or route
            regressions[name]=dict(unconstrained=evaluate(decisions),explicit_time_control=evaluate(guarded))
    sys.path.insert(0,str(ROOT/'experiments/2b-temporal-router-v1'));from temporal_policy import lexical_route
    controls={}
    for role in ('train','dev','source_confirmation'):
        ss=[s for s in ids if by[s]['role']==role];controls[role]={}
        for name in ('lexical','always_recent','always_full'):
            pp=[1 if (lexical_route(request_by[s]['question'])['route']=='full' if name=='lexical' else name=='always_full') else -1 for s in ss]
            controls[role][name]=agreement([by[s]['signed_route_target'] for s in ss],pp,[by[s]['source_group'] for s in ss])
    if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):raise RuntimeError('fit input changed')
    out.mkdir();save(out/'models.json',models);save(out/'predictions.json',predictions)
    save(out/'analysis.json',dict(teacher_label_agreement_only=metrics,paired_teacher_route_diagnostic=contrasts,
        teacher_controls=controls,opened120_frozen_answer_regression=regressions,opened120_baselines=baselines(),
        input_SHA256=hashes,new_training_N=195,parameters=33,head_search=False,threshold_search=False,
        primary='repaired195/recent_video',new_QA_answers=0,base_updates=0,API=0,GPU=0,
        no_default_change=True,independent_benchmark_gain_not_established=True,final_data_repair_round=True))
    print('Fixed33 comparison complete; no new QA answers, no default change')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--feature-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.feature_run,a.out)
