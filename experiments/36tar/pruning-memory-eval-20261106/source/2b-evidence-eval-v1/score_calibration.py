"""Evaluate judge calibration against frozen, single-assistant reference labels."""
import json
from pathlib import Path
def evaluate(rows):
    assert len({r['request_id'] for r in rows})==len(rows)
    original=[r for r in rows if r['variant']=='original']
    sufficient=[r for r in original if r['expected_support']=='sufficient']
    insufficient=[r for r in original if r['expected_support']=='insufficient']
    uncertain=[r for r in original if r['expected_support']=='unverifiable']
    assert (len(sufficient),len(insufficient),len(uncertain))==(8,8,4)
    true_positive=sum(r['support']['sufficiency']=='sufficient' for r in sufficient)
    false_positive=sum(r['support']['sufficiency']=='sufficient' for r in insufficient)
    abstain_correct=sum(r['support']['sufficiency']=='unverifiable' for r in uncertain)
    source={r['calibration_id']:r for r in original};stability={}
    for variant in ('blind_repeat','paraphrase'):
        variants=[r for r in rows if r['variant']==variant];assert len(variants)==10
        reversals=0;support_changes=0
        for r in variants:
            ref=source[r['calibration_id']]
            reversals+=set((r['fact']['verdict'],ref['fact']['verdict']))=={'supported','contradicted'}
            support_changes+=r['support']['sufficiency']!=ref['support']['sufficiency']
        stability[variant]=dict(substantive_fact_reversals=reversals,support_rating_changes=support_changes,pairs=10)
    passed=true_positive>=7 and false_positive<=1 and abstain_correct>=3 and all(x['substantive_fact_reversals']<=1 for x in stability.values())
    return dict(passed_startup_check_only=passed,sufficient_detected=true_positive,insufficient_false_accepts=false_positive,
        unverifiable_detected=abstain_correct,fact_agreement=sum(r['fact']['verdict']==r['expected_fact'] for r in original),
        fact_denominator=20,stability=stability,reference_review='single assistant, not independent human gold',
        general_judge_reliability_proven=False)
def self_check():
    rows=[]
    for i in range(20):
        support='sufficient' if i<8 else 'insufficient' if i<16 else 'unverifiable'
        fact='supported' if i<8 or i==15 else 'contradicted' if i<16 else 'unverifiable'
        rows.append(dict(request_id=f'v{i}',calibration_id=f'c{i}',variant='original',expected_support=support,expected_fact=fact,
                         fact=dict(verdict=fact),support=dict(sufficiency=support)))
    for variant in ('blind_repeat','paraphrase'):
        rows.extend(dict(r,request_id=variant+r['request_id'],variant=variant) for r in rows[:10])
    assert evaluate(rows)['passed_startup_check_only']
    rows[8]['support']=dict(sufficiency='sufficient');rows[9]['support']=dict(sufficiency='sufficient')
    assert not evaluate(rows)['passed_startup_check_only']
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args();self_check()
    if a.run:
        assert json.loads((a.run/'completed.json').read_text())['views']==40
        result=evaluate(json.loads((a.run/'scores.json').read_text()))
        (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    else:print('Synthetic calibration scorer controls passed; no real judge result')
