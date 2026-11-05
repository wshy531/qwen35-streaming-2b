"""Fixed grouped OOF fitting; labels live offline, never in feature extraction."""
import argparse
from pathlib import Path
import sys
import numpy as np
from linear_head import BenefitHead, validate_features
from prepare import ROOT, read, save, digest

def development_decision(video, text, base):
    def group(policy,cohort,role):
        return next(r['correct'] for r in policy['groups'] if r['cohort']==cohort and r['role']==role)
    def source_counts(policy):
        counts={}
        for r in policy['by_source']:counts[r['source']]=counts.get(r['source'],0)+r['correct']
        return counts
    vc,bc=source_counts(video),source_counts(base['lexical'])
    improved=sum(vc[s]>bc[s] for s in vc)
    gates=dict(
        official_not_below_full=group(video,'official102_reference','unpartitioned_official_reference')>=124,
        official_beats_lexical=group(video,'official102_reference','unpartitioned_official_reference')>121,
        current_not_below_lexical=group(video,'transition6_development','current')>=50,
        old_not_below_full=group(video,'transition6_development','old')>=19,
        joint_not_below_lexical=sum(r['both_old_and_current_correct'] for r in video['joint_old_current'])>=12,
        two_sources_net_improve_over_lexical=improved>=2,
        beats_no_video_total=sum(r['correct'] for r in video['groups'])>sum(r['correct'] for r in text['groups']))
    return dict(gates=gates,passed=all(gates.values()),source_net_improvement_count=improved,
        action='prepare_new_source_validation' if all(gates.values()) else 'stop_configuration_no_sweep')

def oof(features, labels):
    x=validate_features(features)
    if len(x)!=len(labels) or len({r['task_id'] for r in labels})!=len(labels):
        raise ValueError('one feature per unique task required')
    scores=np.full(len(x),np.nan);folds=[]
    for held in range(3):
        train=np.array([i for i,r in enumerate(labels) if r['fold']!=held])
        test=np.array([i for i,r in enumerate(labels) if r['fold']==held])
        if not len(train) or not len(test):raise ValueError('empty fold')
        tr_groups={labels[i]['leakage_group'] for i in train}
        te_groups={labels[i]['leakage_group'] for i in test}
        if tr_groups & te_groups:raise ValueError('source leakage')
        head=BenefitHead().fit(x[train],[labels[i]['target_gain'] for i in train],
            [labels[i]['fit_weight'] for i in train],[labels[i]['leakage_group'] for i in train])
        scores[test]=head.predict_gain(x[test])
        folds.append(dict(heldout_fold=held,train_ids=[labels[i]['task_id'] for i in train],
            heldout_ids=[labels[i]['task_id'] for i in test],coefficient=head.coefficient.tolist(),
            mean=head.mean.tolist(),scale=head.scale.tolist()))
    if not np.isfinite(scores).all():raise ValueError('missing OOF prediction')
    return scores,folds

def run(audit, features, output):
    labels=read(audit/'policy_targets_LOCAL_ONLY.json')
    expected=[r['task_id'] for r in labels]
    with np.load(features,allow_pickle=False) as data:
        if set(data.files)!={'ids','recent_video','no_video'}:
            raise ValueError('feature archive schema mismatch')
        if data['ids'].tolist()!=expected:
            raise ValueError('feature IDs/order differ from locked task labels')
        arrays={mode:data[mode].copy() for mode in ('recent_video','no_video')}
    sys.path.insert(0,str(ROOT/'experiments/2b-recent-evidence-v1'))
    from evidence_analysis import evaluate,baselines
    policies={};predictions={};models={}
    for mode,x in arrays.items():
        scores,folds=oof(x,labels)
        decisions={(r['task_id'],shift):('full' if scores[i]>0 else 'recent')
            for i,r in enumerate(labels) for shift in range(5)}
        policies[mode]=evaluate(decisions)
        predictions[mode]=[dict(task_id=r['task_id'],fold=r['fold'],gain=float(scores[i]),
            route=decisions[(r['task_id'],0)]) for i,r in enumerate(labels)]
        models[mode]=folds
    base=baselines()
    result=dict(status='retrospective_fixed_answer_selection',policies=policies,baselines=base,
        development_decision=development_decision(policies['recent_video'],policies['no_video'],base),
        predictions=predictions,models=models,features_sha256=digest(features),
        labels_sha256=digest(audit/'policy_targets_LOCAL_ONLY.json'),
        all_120_tasks_evaluated=True,rotations_not_independent=True,
        per_task_same_route_for_rotations=True,new_QA_answers=0,
        formal_generalization_claim=False,default_enable=False)
    save(output,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.audit,a.features,a.output)
