import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/anchor_merge_development_20261103_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
P.mkdir(exist_ok=False)
old=R/'runs/learned_prune_main_prepare_20261101_01'
qs=[q for q in read(old/'eval.json') if q['role']=='development'];assert len(qs)==345
vids={q['video_id'] for q in qs};assert len(vids)==40
sources=[s for s in read(old/'sources.json') if s['video_id'] in vids]
audit=[s for s in read(R/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows'] if s['video_id'] in vids]
baseline=[r for r in read(R/'runs/hard_mask_main_prepare_20261102_01/baseline_answers.json') if r['id'] in {q['id'] for q in qs}]
assert len(baseline)==2070
pairs=sum((len(s['samples'])+1)//2 for s in audit)
values={'questions.json':qs,'sources.json':sources,'samples.json':audit,'baseline.json':baseline,
 'protocol.json':dict(arms=['full','fixed12','heuristic11','anchor_merge11'],questions=345,sources=40,
 rotations=[0,2],formal_answers=2760,warmup_answers=4,max_generated_tokens=22112,
 vision_calls=4*pairs,real_pairs=pairs,pretrained_loads=1,training=False,minutes=60,
 max_results_MiB=96,scoring_after_layer=11,deletion_after_layer=12,keep=54,
 old_endpoint_controls=2070,role='opened development; not blind confirmation',
 next_panel='entire comprehensive300, separate explanation protocol; do not choose by dev wins')}
for name,value in values.items():(P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(values['protocol.json']))
