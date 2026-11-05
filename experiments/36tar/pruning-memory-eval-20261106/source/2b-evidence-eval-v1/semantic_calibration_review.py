"""Semantic audit after user accepted format flexibility. Preserve original gate."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[2];RUN=R/'runs/evidence-judge-calibration-resume-20261102-01'
rows=json.loads((RUN/'scores.json').read_text(encoding='utf-8'))
original=[r for r in rows if r['variant']=='original'];assert len(original)==20
definite=[r for r in original if r['expected_support']!='unverifiable']
unknown=[r for r in original if r['expected_support']=='unverifiable']
assert len(definite)==16 and len(unknown)==4
result=dict(original_startup_gate_passed=False,original_gate_preserved=True,
    factual_verdict_agreement=dict(correct=sum(r['fact']['verdict']==r['expected_fact'] for r in original),denominator=20),
    support_on_definite_cases=dict(correct=sum(r['support']['sufficiency']==r['expected_support'] for r in definite),denominator=16),
    unknown_facts=dict(correctly_unverifiable=sum(r['fact']['verdict']=='unverifiable' for r in unknown),denominator=4,
        support_ratings={r['calibration_id']:r['support']['sufficiency'] for r in unknown},
        interpretation='Insufficient support is compatible with an unverifiable premise; old support labels conflated these questions.'),
    formatting_is_not_content_accuracy=True,independent_gold=False,new_API_calls=0,
    limits=['3 sources only','single assistant reference review','single-claim calibration does not validate complex claim decomposition',
            'no proof of internal reasoning faithfulness or general judge accuracy'])
(RUN/'semantic_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
