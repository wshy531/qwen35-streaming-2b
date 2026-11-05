import json
from pathlib import Path
from response_v2 import parse
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01';RUN=R/'runs/evidence-profile-220065'
rows=[json.loads(x) for x in (RUN/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
qs={q['id']:q for q in json.loads((P/'profile.json').read_text(encoding='utf-8'))}
labels=json.loads((P/'labels_nextqa.json').read_text());changes=[]
assert parse('A. one\nEvidence: a child',5,'B')['original_selected_index']==0
assert not parse('The answer could be A or B',5,'B')['answer_parseable']
assert not parse('Evidence: A. cat',5,'B')['answer_parseable']
assert not parse('E. invalid',4,'B')['answer_parseable']
for row in rows:
    prior=row['original_selected_index'];new=parse(row['text'],len(qs[row['id']]['options']),row['protocol'])
    row.update(new)
    if prior!=row['original_selected_index']:changes.append(dict(id=row['id'],arm=row['arm'],prior=prior,new=row['original_selected_index']))
summary={}
for arm in ('full','fixed12'):
    for proto in ('A','B'):
        rr=[r for r in rows if r['arm']==arm and r['protocol']==proto]
        summary[f'{arm}/{proto}']=dict(content_correct=sum(r['original_selected_index']==labels[r['id']] for r in rr),questions=len(rr),
            parseable=sum(r['answer_parseable'] for r in rr),explanations=sum(r['explanation_present'] for r in rr),
            support_fields=sum(bool(r['support']) for r in rr),max_length_reached=sum(len(r['generated_ids'])==(8 if proto=='A' else 128) for r in rr))
(RUN/'answers_reparsed_v2.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
result=dict(summary=summary,changes=changes,original_answers_unchanged=True,GPU=0,API=0)
(RUN/'reparse_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
