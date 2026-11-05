from followup import *
import json
assert not OUT.exists()
qs=read(P/'questions.json');assert len(qs)==150
rows=[json.loads(s) for s in (R/'runs/protect-stream-merged-220270-220293/answers.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows)==600 and len({(x['id'],x['arm']) for x in rows})==600
controls=[q['id'] for q in sorted(qs,key=lambda q:(q['source_id'],q['query_time_s'],q['id']))[:16]]
expected={}
for row in rows:
    if row['arm'] in ('joint','recent'):
        expected.setdefault(row['id'],{})[row['arm']]={k:row[k] for k in ('generated_ids','bounded_context_ids_sha256')}
        if 'position_sha256' in row:expected[row['id']][row['arm']]['position_sha256']=row['position_sha256']
OUT.mkdir()
save(OUT/'manifest.json',dict(questions=qs,control_ids=controls,expected=expected,
    question_sha256=sha(P/'questions.json'),head_sha256=sha(HEAD_PATH),
    source_answers_sha256=sha(R/'runs/protect-stream-merged-220270-220293/answers.jsonl'),
    methods=['API33_LS','A06_joint'],formal_answers=300,endpoint_controls=32,warmup_answers=2,
    max_generated_tokens=5344,max_feature_extractions=150,vision_pairs=25815,
    proposed_gpu_minutes=110,approved=False))
print('prepared 150 questions; no GPU submitted')
