"""Mechanism checks with synthetic features only. Never loads model weights."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import numpy as np
from contract import supervision, source_groups, runtime_request, normalized_question
from linear_head import BenefitHead
from offline_fit import oof, development_decision
from prepare import ROOT, INPUT, read, digest, save

def check(audit_dir, output):
    requests = read(INPUT / 'requests.json')
    tasks = read(INPUT / 'evaluation_tasks_LOCAL_ONLY.json')
    rows = read(INPUT / 'frozen_branch_outputs_LOCAL_ONLY.json')
    labels = read(audit_dir / 'policy_targets_LOCAL_ONLY.json')
    metadata = {s['name']: s for s in read(INPUT / 'sources.json')}
    result = {}
    def require(name, value):
        if not value:
            raise AssertionError(name)
        result[name] = True
    g = source_groups(requests, metadata)
    require('split_invariant_to_request_order', g == source_groups(list(reversed(requests)), metadata))
    byquestion = {}
    for r in requests:
        key = normalized_question(r['question'])
        if key in byquestion:
            require('exact_question_duplicates_share_fold', g[r['id']]['fold'] == byquestion[key])
        byquestion[key] = g[r['id']]['fold']
    require('all_120_tasks_and_600_rows_retained', len(labels) == len(tasks) == 120 and len(rows) == 600)
    require('three_folds_nonempty', {r['fold'] for r in labels} == {0,1,2})
    require('unknown_support_labels_stay_null', all(r['visual_sufficiency_label'] is None for r in labels))
    require('both_wrong_never_forced_to_history', sum(r['fit_weight']==0 for r in labels)==42 and
            all(r['target_gain']==0 for r in labels if r['fit_weight']==0))
    require('rotations_aggregate_not_expand_examples', len(labels)==120 and
            sum(r['recent_correct'] for r in labels)==161 and sum(r['full_correct'] for r in labels)==182)
    for held in range(3):
        tr = {r['leakage_group'] for r in labels if r['fold'] != held}
        te = {r['leakage_group'] for r in labels if r['fold'] == held}
        require(f'fold_{held}_group_disjoint', not tr & te)
    poisoned = dict(requests[0], correct_answer='LEAK', role='LEAK', target_gain=1,
                    recent_correct=5, full_correct=0, fold=2, source_name='LEAK')
    view = runtime_request(poisoned)
    require('feature_request_allowlist_excludes_labels_roles_sources_options',
            set(view)=={'id','question','recent_frames','query_time_s'} and 'LEAK' not in json.dumps(view))
    sample = [dict(shift=i,recent=dict(correct=False),full=dict(correct=False)) for i in range(5)]
    require('synthetic_all_failure_zero_weight', supervision(sample)['fit_weight']==0)
    sample[0]['recent']['correct']=True
    sample[1]['full']['correct']=True
    require('opposing_rotations_not_majority_relabeled', supervision(sample)['target_gain']==0 and
            supervision(sample)['opposing_rotation_preferences'])
    malformed = copy.deepcopy(sample); malformed[-1]['shift']=0
    try:
        supervision(malformed)
    except ValueError:
        require('duplicate_rotation_rejected', True)
    else:
        require('duplicate_rotation_rejected', False)
    # Synthetic signal: two train sources plus an explicit zero-weight failure.
    rng=np.random.default_rng(7)
    x=rng.normal(size=(12,2048)).astype(np.float32)
    y=np.array([-.8,.6,-.4,.8,-.6,.4,-.2,1.,-.6,.8,.4,0.])
    weights=np.array([1.]*11+[0.]); groups=['a']*6+['b']*6
    head=BenefitHead().fit(x,y,weights,groups)
    require('only_33_parameters_fit', head.coefficient.shape==(33,))
    require('synthetic_fit_numerically_finite', np.isfinite(head.predict_gain(x)).all())
    changed=x.copy(); changed[-1]=1e6
    h2=BenefitHead().fit(changed,y,weights,groups)
    require('zero_weight_inputs_do_not_change_fit', np.allclose(head.coefficient,h2.coefficient,atol=1e-11) and
            np.allclose(head.mean,h2.mean,atol=1e-11))
    repeated_x=np.concatenate((x[:6],x[:6],x[6:]))
    repeated_y=np.concatenate((y[:6],y[:6],y[6:]))
    repeated_w=np.concatenate((weights[:6],weights[:6],weights[6:]))
    h3=BenefitHead().fit(repeated_x,repeated_y,repeated_w,['a']*12+['b']*6)
    require('group_mass_not_multiplied_by_replication', np.allclose(head.coefficient,h3.coefficient,atol=1e-10))
    frozen=head.coefficient.copy(); head.predict_gain(rng.normal(size=(8,2048)))
    require('evaluation_never_refits_statistics', np.array_equal(frozen,head.coefficient))
    invalid=x.copy(); invalid[0,0]=np.nan
    try:
        BenefitHead().fit(invalid,y,weights,groups)
    except ValueError:
        require('nonfinite_features_rejected', True)
    else:
        require('nonfinite_features_rejected', False)
    synthetic_features=rng.normal(size=(len(labels),2048)).astype(np.float32)
    base_scores,models=oof(synthetic_features,labels)
    changed_labels=copy.deepcopy(labels)
    for r in changed_labels:
        if r['fold']==0:r['target_gain']=-r['target_gain']
    changed_scores,_=oof(synthetic_features,changed_labels)
    held0=np.array([r['fold']==0 for r in labels])
    require('heldout_labels_do_not_change_own_predictions', np.array_equal(base_scores[held0],changed_scores[held0]))
    require('all_tasks_receive_single_out_of_fold_prediction', np.isfinite(base_scores).all() and
            sum(len(m['heldout_ids']) for m in models)==120)
    import sys
    sys.path.insert(0,str(ROOT/'experiments/2b-recent-evidence-v1'))
    from evidence_analysis import baselines
    reference=baselines()
    require('always_full_fails_current_protection_gate',
            not development_decision(reference['full'],reference['recent'],reference)['passed'])
    require('lexical_cannot_pass_beat_itself_gate',
            not development_decision(reference['lexical'],reference['recent'],reference)['passed'])
    sources=list(Path(__file__).resolve().parent.glob('*.py'))
    receipt=dict(checks=result, passed=all(result.values()), numpy=np.__version__,
        real_model_loaded=False, real_features_available=False, real_head_fit=False,
        synthetic_head_fit_only=True, GPU_jobs=0, new_answers=0,
        files=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in sorted(sources)])
    save(output, receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check(a.audit,a.output)
