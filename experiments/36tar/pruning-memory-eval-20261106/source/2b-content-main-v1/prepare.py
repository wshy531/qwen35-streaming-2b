"""Freeze unchanged R2 inputs, balanced metadata-only shards and old exact references."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];PREP=ROOT/'runs/content_main_prepare_20261101_01'
MODES=['full','post_fixed','mid_fixed','post_content','mid_content']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    PREP.mkdir(exist_ok=True)
    if (PREP/'protocol.json').exists():raise FileExistsError('retain frozen protocol')
    src=ROOT/'runs/midprune_quality_prepare_20261026_01/inputs'
    qs=read(src/'requests.json');expected=read(src/'expected_prefixes.json');old=read(src/'protocol.json')
    for name in ('requests.json','videos.json','expected_prefixes.json'):(PREP/name).write_bytes((src/name).read_bytes())
    vids={q['video_id'] for q in qs};costs={v:max(expected[q['question_id']]['pairs'] for q in qs if q['video_id']==v) for v in vids}
    shards=[[],[]];loads=[0,0]
    for v in sorted(vids,key=lambda v:(-costs[v],v)):
        i=min(range(2),key=lambda i:(loads[i],i));shards[i].append(v);loads[i]+=costs[v]
    tasks=[dict(name='quality_'+str(i),kind='quality',modes=MODES,videos=sorted(s),minutes=40) for i,s in enumerate(shards)]
    for i,block in enumerate(old['cost_blocks']):
        order=MODES[i%5:]+MODES[:i%5]
        tasks.extend(dict(name=f'cost_b{i}_{m}',kind='cost',modes=[m],videos=block,minutes=35) for m in order)
    assert sum(costs.values())==14678 and sum(len([q for q in qs if q['video_id'] in s]) for s in shards)==478
    refs={}
    source=ROOT/'runs/midprune_quality_prepare_20261026_01/GPU_results_218670/quality/answers.jsonl'
    mapping={'full108_8':'full','post_odd54_8':'post_fixed','mid_odd54_8':'mid_fixed'}
    for line in source.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['arm'] in mapping:
            refs[r['question_id']+'|'+mapping[r['arm']]]={k:r[k] for k in ('generated_ids','text','question_suffix_sha256','bounded_context_ids_sha256','visual_tokens')}
    assert len(refs)==1434
    save(PREP/'references.json',refs);save(PREP/'tasks.json',tasks)
    protocol=dict(videos=128,questions=478,modes=MODES,shard_pairs=loads,quality_formal_answers=2390,
        quality_warmups=10,cost_formal_answers=285,cost_warmups=20,maximum_generated_tokens=43280,
        max_vision_calls=73390+9550+30,pretrained_loads=22,cost_source_count=12,cost_question_count=57,
        per_method_processes=4,cost_blocks=old['cost_blocks'],cost_order='four disjoint blocks; cyclic five-arm order, imperfect full balance reported',
        decoder='auto4',fps=.5,canvas=[288,384],dtype='FP32',pair_coverage=8,
        full_tokens=864,pruned_tokens=432,quality_development=True,
        native_reference='not full native video; complete visual means eight-pair window',
        candidate_frozen_no_parameter_search=True,baseline_reference_rows=1434,
        original_prefix_identity_keys=old['source_identity_keys'],
        cost_excludes='media SHA, model load, warmups; per-source total includes decode through QA and audits',
        no_new_API_download_or_training=True,no_auto_retry=True)
    save(PREP/'protocol.json',protocol)
    print(json.dumps(protocol))
if __name__=='__main__':main()
