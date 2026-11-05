import json
from pathlib import Path
from response_v3 import parse
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
labels=read(P/'labels_nextqa.json');questions={q['id']:q for q in read(P/'profile.json')}
assert parse('A. pat it\nAnswer: D. standing',5,'B')['original_selected_index']==0
assert parse('A. pat it\nAnswer: D. standing',5,'B')['answer_conflict']
assert parse('Answer: B\nAnswer: A',5,'B')['original_selected_index']==1
assert not parse('Evidence: A. cat',5,'B')['answer_parseable']
tables={}
for run_name in ('evidence-profile-220065','evidence-format-220066'):
    run=R/'runs'/run_name;rows=[json.loads(x) for x in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    changes=[]
    for r in rows:
        old=r['original_selected_index'];r.update(parse(r['text'],len(questions[r['id']]['options']),'A' if r['protocol']=='A' else 'B'))
        if old!=r['original_selected_index']:changes.append(dict(id=r['id'],arm=r['arm'],old=old,new=r['original_selected_index']))
    table={}
    for arm in ('full','fixed12'):
        for proto in sorted({r['protocol'] for r in rows}):
            rr=[r for r in rows if r['arm']==arm and r['protocol']==proto]
            table[f'{arm}/{proto}']=dict(questions=len(rr),content_correct=sum(r['original_selected_index']==labels[r['id']] for r in rr),
                answer_parseable=sum(r['answer_parseable'] for r in rr),labeled_evidence=sum(r['explanation_present'] for r in rr),
                labeled_support=sum(bool(r['support']) for r in rr),answer_conflicts=sum(r['answer_conflict'] for r in rr))
    result=dict(content=table,parser_changes=changes,first_explicit_answer_is_locked=True,API=0,GPU=0)
    (run/'answers_reparsed_v3.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (run/'analysis_v3.json').write_text(json.dumps(result,indent=2)+'\n');tables[run_name]=result
print(json.dumps(tables))
