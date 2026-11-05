import json
from pathlib import Path
from response import *
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
cases=[('Answer: B\nEvidence: A cat.\nSupport: It is a cat.',5,'B',1,2),
       ('Answer: A\nEvidence: unseen\nAnswer: C\nSupport: unknown',4,'B',0,0),
       ('B',5,'A',2,3),('A or B',4,'A',0,None),('Answer: E',4,'B',0,None),
       ('Evidence: A bird\nSupport: B is likely',5,'B',0,None)]
for text,n,protocol,shift,expected in cases:assert parse(text,n,protocol,shift)['original_selected_index']==expected
assert parse(cases[1][0],4,'B')['answer_conflict']
assert not parse('Answer: A',5,'B')['explanation_present']
count=0
for q in json.loads((P/'panel.json').read_text(encoding='utf-8')):
    assert not {'answer_index','answer_text','correct'}.intersection(q)
    for shift in (0,2):
        for i in range(len(q['options'])):
            r=parse(f'Answer: {chr(65+i)}\nEvidence: x\nSupport: y',len(q['options']),'B',shift)
            assert q['options'][r['original_selected_index']]==(q['options'][shift:]+q['options'][:shift])[i];count+=1
material=dict(material_id='test',frames=[dict(pts_s=1.,id='f1')],complete=True,answer_index='LEAK',model='LEAK')
response=parse('Answer: A\nEvidence: visible cup\nSupport: explains answer',4,'B')
assert set(fact_payload(material,response))=={'material_id','frames','complete','observations'}
assert 'LEAK' not in json.dumps(fact_payload(material,response))
result=dict(parser_boundary_cases=len(cases)+2,option_mapping_checks=count,request_allowlist=True,GPU=0,API=0)
(P/'CPU_protocol.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
