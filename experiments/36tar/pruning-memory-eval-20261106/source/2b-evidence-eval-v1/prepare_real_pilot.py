"""Choose six source-separated real questions by metadata, not answer outcomes."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
panel=[q for q in json.loads((P/'panel.json').read_text(encoding='utf-8')) if q['dataset']=='NExT-QA']
selected=[];used_types=set();used_sources=set()
for q in sorted(panel,key=lambda q:hashlib.sha256(('real-grounding-pilot-v1:'+q['id']).encode()).hexdigest()):
    if q['source_id'] in used_sources or q['type'] in used_types:continue
    selected.append(q);used_sources.add(q['source_id']);used_types.add(q['type'])
    if len(selected)==6:break
assert len(selected)==6
answers=[json.loads(x) for x in (R/'runs/evidence-nextqa-220067/answers.jsonl').read_text(encoding='utf-8').splitlines()]
cases=[]
for q in selected:
    for arm in ('full','fixed12','heuristic11'):
        response=next(r for r in answers if r['id']==q['id'] and r['arm']==arm and r['protocol']=='B' and r['shift']==0)
        cases.append(dict(opaque_id=f'real-{len(cases):02d}',question=q['question'],options=q['options'],source_id=q['source_id'],
            query_time_s=q['query_time_s'],question_id=q['id'],type=q['type'],response=response))
(P/'real_pilot_LOCAL_ONLY.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(cases=len(cases),questions=6,sources=6,types=sorted(used_types),selection_without_scores=True,API=0)))
