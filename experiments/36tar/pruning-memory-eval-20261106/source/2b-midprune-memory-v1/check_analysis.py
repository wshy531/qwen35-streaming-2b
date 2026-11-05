"""Known synthetic outcomes test role denominators, paired loss and malformed-answer accounting."""
from analyze import *
labels=read(P/'labels_LOCAL_ONLY.json');rows=[]
for view in ('full','fixed12','heuristic11','mask_ce'):
    policies=['M0','M2','A06','LS']+([] if view=='full' else ['M0_same8'])
    for policy in policies:
        for i,l in labels.items():
            selected=l['correct_index'];parsed=True
            if view=='mask_ce' and policy=='LS' and i in ('T01:early:current','T01:late:old'):selected=None;parsed=False
            rows.append(dict(id=i,view=view,policy=policy,original_selected_index=selected,answer_parseable=parsed,
                read_state_unchanged=True,branch_released=True,bank_status=dict(tokens=432,visual_budget=864)))
r=analyze(rows,labels);d=r['methods']['mask_ce/LS']
assert d['current']['questions']==12 and d['current']['option_content_correct']==11
assert d['old']['questions']==6 and d['old']['option_content_correct']==5
assert d['late_both_correct']==5 and len(d['versus_full_same_policy']['lost'])==2
for bad in (rows[:-1],rows+[rows[0]]):
    try:analyze(bad,labels)
    except AssertionError:pass
    else:raise AssertionError('incomplete or duplicate score accepted')
print(json.dumps(dict(passed=True,synthetic_rows=len(rows),known_role_and_pair_scores=True,incomplete_duplicate_rejected=True)))
