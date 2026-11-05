import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/hard_mask_main_prepare_20261102_01'
O=R/'runs/learned_prune_main_prepare_20261101_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(P/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
P.mkdir(exist_ok=True)
train=read(O/'train.json');evaluation=read(O/'eval.json')
audit={r['video_id']:r for r in read(R/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
def pairs(q):return (len(audit[q['video_id']]['samples'])+1)//2
def one_per_source(rows,salt,n):
    selected=[];seen=set()
    for q in sorted(rows,key=lambda q:hashlib.sha256((salt+q['id']).encode()).hexdigest()):
        if q['video_id'] not in seen:selected.append(q);seen.add(q['video_id'])
        if len(selected)==n:break
    assert len(selected)==n
    return selected
gates=one_per_source(evaluation,'hard-mask-reference-v1:',12)
probes=one_per_source(train,'hard-mask-training-probe-v1:',24)
save('reference_questions.json',gates);save('training_probes.json',probes)
old=[json.loads(line) for line in (R/'runs/learned-prune-main-220024/answers.jsonl').read_text(encoding='utf-8').splitlines()]
baseline=[r for r in old if r['arm'] in ('full','fixed12','heuristic11')]
assert len(baseline)==3912
save('baseline_answers.json',baseline)
reference=[r for r in baseline if r['id'] in {q['id'] for q in gates}]
assert len(reference)==72
save('reference_subset.json',reference)
train_vids={q['video_id']:q for q in train};eval_vids={q['video_id']:q for q in evaluation}
vision=2*sum(pairs(q) for q in train)+sum(pairs(q) for q in train_vids.values())+3*sum(pairs(q) for q in eval_vids.values())+3*sum(pairs(q) for q in gates)+3*sum(pairs(q) for q in probes)
spec=dict(training_questions=1237,training_sources=150,updates=2474,teacher_forwards=1237,train_probe_forwards=72,
    eval_questions=652,eval_sources=78,eval_answers=3912,reference_answers=72,total_answers=3984,
    max_generated_tokens=31872,vision_calls=vision,epochs_per_arm=1,arms=['mask_initial','mask_ce','mask_ce_kl'],
    baseline_reuse_arms=['full','fixed12','heuristic11'],eval_rotations=[0,2],baseline_job='220024',
    checkpoint_primary='final epoch endpoint; intermediate diagnostics only',trainable_parameters=69697,
    dtype='float32',model_loads=1,base_updates=0,keep=54,total_groups=108,lr=1e-4,clip=1.,kl_weight=.5,
    max_minutes=180,results_MiB=128,logs_MiB=64,no_retry=True,no_requeue=True,
    estimated_cost='about 2 to 2.5 hours from 3-clip profile plus previous QA; reserve 3h hard cap, not measured full run')
save('protocol.json',spec);print(json.dumps(spec))
