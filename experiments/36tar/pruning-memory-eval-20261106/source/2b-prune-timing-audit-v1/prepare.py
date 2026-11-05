from audit_common import ROOT,PREP,MODES,read,save
PREP.mkdir(exist_ok=False)
cases=read(ROOT/'runs/prune_failure_inputs_20261103_01/cases.json')
ids={q['id'] for q in cases};assert len(ids)==6
requests=[q for q in read(ROOT/'runs/content_confirm_prepare_20261101_01/requests.json') if q['id'] in ids]
assert len(requests)==6 and len({q['media_name'] for q in requests})==6
requests.sort(key=lambda q:next(i for i,c in enumerate(cases) if q['id']==c['id']))
assert not any(set(q)&{'label','answer','correct','outcomes'} for q in requests)
save(PREP/'requests.json',requests)
save(PREP/'media.json',[m for m in read(ROOT/'runs/content_confirm_prepare_20261101_01/media.json') if m['name'] in {q['media_name'] for q in requests}])
old=[x for x in read(ROOT/'runs/content_confirm_prepare_20261101_01/confirmation_analysis.json')['scored'] if x['task_id'] in ids and x['arm'] in ['full','mid_fixed','mid_content']]
assert len(old)==72 and len({(x['task_id'],x['arm'],x['shift']) for x in old})==72
save(PREP/'reference_choices.json',[{k:x[k] for k in ('task_id','arm','shift','choice')} for x in old])
assert all(q['outcomes']['full']['correct'] for q in cases)
save(PREP/'labels_LOCAL_ONLY.json',{q['id']:q['outcomes']['full']['choice'] for q in cases})
manifest=read(ROOT/'runs/prune_failure_inputs_20261103_01/manifest.json')
save(PREP/'sample_contract.json',[dict(id=x['question']['id'],pairs=x['pairs_received'],state=x['state'],frames=x['groups']['actual_last8_pairs']) for x in manifest])
assert sum(x['pairs_received'] for x in manifest)==513
save(PREP/'protocol.json',dict(questions=6,sources=6,arms=MODES,rotations=[0,1,2,3],formal_answers=120,warmup_answers=5,
    max_generated_tokens=1000,visual_calls=2565,pretrained_loads=1,minutes=30,GPU=1,cpus=4,memory_GiB=24,
    output_MiB=48,partition='g078t',comment='gx_llm',training=False,API=False,download=False,
    selection_ID_equality='every pair, fixed and content; abort on mismatch',old_choice_regression=72,
    evaluation_role='six selected failures, mechanism only, NOT representative accuracy',
    existing_evidence='478 question five-path comparison already exists; this is complementary case diagnosis',
    interpretation='first original order then all rotations; separate prediction/count evidence caveats',
    next_quality_panels={'development':345,'opened_regression':307,'comprehensive':300},
    new_source_confirmation='must be frozen before candidate results; not yet prepared',
    stop='one diagnostic and at most one mechanism-supported repair; no layer/budget sweep'))
print('Six frozen sources, 120 answers, 72 old-choice controls prepared')
