"""Reuse all existing tasks, never filter on candidate outcomes."""
from experiment import *
from collections import Counter
P.mkdir(exist_ok=False)
old=R/'runs/resolution_cost_prepare_20261025_01/inputs'
for name in ('requests.json','videos.json','expected_mid_prefixes.json'):
    (P/name).write_bytes((old/name).read_bytes())
paired=R/'runs/recent_detail_prepare_20261103_01'
for name in ('questions.json','sources.json'):(P/('paired_'+name)).write_bytes((paired/name).read_bytes())
qs=read(P/'requests.json');sources=read(P/'videos.json');pq=read(P/'paired_questions.json');ps=read(P/'paired_sources.json')
assert len(qs)==478 and len(sources)==128 and len(pq)==18 and len(ps)==5
raw=R/'runs/resolution_cost_prepare_20261025_01/GPU_results_218537/quality_mid/answers.jsonl'
baseline=[{k:r[k] for k in ('question_id','generated_ids','text','bounded_context_ids_sha256','visual_tokens')} for r in map(json.loads,raw.read_text(encoding='utf-8').splitlines()) if r['arm']=='LS_mid108']
assert len(baseline)==478;save(P/'baseline.json',baseline)
expected=read(P/'expected_mid_prefixes.json')
by_source={v['video_id']:max(expected[q['question_id']]['pairs'] for q in qs if q['video_id']==v['video_id']) for v in sources}
paired_pairs=sum(len(s['sampled_trace'])//2 for s in ps)
numeric=sorted(q['question_id'] for q in qs)[::60][:8]
plan=dict(arms=list(ARMS),official_questions=478,paired_questions=18,paired_rotations=list(range(5)),
    formal_answers=478*3+18*5*3,warmup_answers=3,numerical_control_answers=8,
    max_answers=1715,max_generated_tokens=1715*16,vision_calls=sum(by_source.values())+paired_pairs,
    numeric_ids=numeric,old_joint_controls=478,canvas=[288,384],fps=.5,
    recent_visual_budget=432,joint_visual_budget=864,bank='LS_mid108',
    paired_media_root=read(paired/'protocol.json')['media_root'],
    paired_sampling='Inherited actual 1 FPS trace; official subset 0.5 FPS; separate reports',
    question_categories=dict(Counter(q['question_category'] for q in qs)),
    source_file_overlap=sorted({s['name'].removesuffix('.mp4') for s in ps}&{v['video_id'] for v in sources}),
    budget=dict(partition='g078t',comment='gx_llm',GPU=1,CPU=4,system_GiB=24,minutes=90,output_MiB=96),
    stop='Inputs/baseline/numerical failure stops, no automatic retry or tuning; preserve every quality result',
    limitations=['478 opened questions test broad retention, not clean current-perception labels',
        '18 diagnostic questions from five familiar sources; five rotations are not independent examples',
        'No new annotations, no API judgement, no bottom-model training, no end-to-end speed claim'])
assert plan['vision_calls']<16000 and plan['formal_answers']==1704
save(P/'protocol.json',plan)
print(json.dumps(plan,ensure_ascii=False))

