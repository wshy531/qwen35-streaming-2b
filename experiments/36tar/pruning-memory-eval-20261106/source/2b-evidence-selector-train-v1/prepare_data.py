"""Reuse reviewed API citations; freeze source-disjoint long-video auxiliary sets."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/evidence_selector_train_prepare_20261029_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def order(s):return hashlib.sha256(('selector-long-v1-20261029:'+s).encode()).hexdigest()

def main():
    if PREP.exists():raise FileExistsError('retain prior preparation')
    data=ROOT/'runs/router-teacher-data-20261029-01/dataset_reviewed_all.json'
    raw=read(data);req=[];supervision={};sources={};summary={}
    sampled=read(ROOT/'runs/router_teacher_prepare_20261029_01/sampled_sources.json')
    sampled={s['id']:s for s in sampled}
    for r in raw:
        if not r['training_weight']:continue
        full=r['history_frames']+r['recent_frames'];assert len(full)%2==0 and len(r['recent_frames'])==8
        pairs=[full[i:i+2] for i in range(0,len(full),2)];hist_count=len(pairs)-4
        cited=set(r['teacher_review']['reference_evidence_ids'])
        positives=[i for i,p in enumerate(pairs[:hist_count]) if cited & {f['frame_id'] for f in p}]
        if r['label']=='HISTORY_NEEDED' and not positives:raise ValueError('no historical cited pair')
        req.append(dict(id=r['id'],source_id=r['source_id'],question=r['question'],query_time_s=r['query_time_s']))
        supervision[r['id']]=dict(role=r['role'],source_group=r['source_group'],needs_history=int(r['label']=='HISTORY_NEEDED'),
            route_known=True,positive_history_indices=positives,uncited_nodes_are_unlabelled=True,
            independent_gold=False,rank_label='API_cited_source_pairs_weak_support',teacher_answer_not_in_input=True)
        sources[r['source_id']]=sampled[r['source_id']]
    assert len(req)==351
    long_req=read(ROOT/'runs/sember_prepare_20261025_01/selection/requests.json')
    long_lab={r['question_id']:r for r in read(ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json')}
    videos=read(ROOT/'runs/sember_prepare_20261025_01/selection/videos.json')
    # All already opened videos are development material; never relabel them unseen.
    candidates=sorted({r['parent_source'] for r in long_req if r['split']=='formal'},key=order)
    split={p:('train' if i<16 else 'dev' if i<20 else 'heldout_development') for i,p in enumerate(candidates[:28])}
    regression=read(ROOT/'runs/evidence_selection_prepare_20261029_01/inputs/requests.json')
    used_parent={t['source_name'].split('_start_')[0] for t in regression}
    assert not set(split)&used_parent
    long_rows=[];long_targets={}
    for r in long_req:
        if r['parent_source'] not in split or r['split']!='formal':continue
        label=long_lab[r['question_id']];a,b=label['answer_start_time'],label['answer_end_time']
        valid=0<=a<=b<=r['question_time']
        tid='long:'+r['question_id']
        long_rows.append(dict(id=tid,source_id=r['video_id'],parent_source=r['parent_source'],
            question=r['question'],query_time_s=r['question_time'],category=r['question_category']))
        long_targets[tid]=dict(role=split[r['parent_source']],route_known=False,intervals=[[a,b]] if valid else [],
            rank_weight=.25,rank_label='official_interval_source_proxy_only',
            recent_sufficiency_unknown=True,compressed_readability_unknown=True,correct_answer_not_used=True)
    picked=[v for v in videos if v['parent_source'] in split]
    parents=defaultdict(set)
    for t in supervision.values():parents[t['role']].add(t['source_group'])
    assert not parents['train']&parents['dev'] and not parents['train']&parents['heldout'] and not parents['dev']&parents['heldout']
    save(PREP/'short_requests.json',req);save(PREP/'short_sources.json',[sources[k] for k in sorted(sources)])
    save(PREP/'supervision_LOCAL_ONLY.json',supervision);save(PREP/'long_requests.json',long_rows)
    save(PREP/'long_sources.json',picked);save(PREP/'long_supervision_LOCAL_ONLY.json',long_targets)
    save(PREP/'excluded_weak_rows_LOCAL_ONLY.json',[dict(id=r['id'],role=r['role'],reasons=r['filter_reasons'],weight=0) for r in raw if not r['training_weight']])
    short_summary={}
    for role in ['train','dev','heldout']:
        s=[r for r in supervision.values() if r['role']==role]
        short_summary[role]=dict(tasks=len(s),parents=len(parents[role]),recent=sum(not r['needs_history'] for r in s),history=sum(r['needs_history'] for r in s),history_citations=sum(bool(r['positive_history_indices']) for r in s))
    save(PREP/'protocol.json',dict(short=short_summary,long_sources_by_role=dict(Counter(split.values())),
        long_questions_by_role=dict(Counter(t['role'] for t in long_targets.values())),
        short_visual_pairs=sum(len(s['full_frames'])//2 for s in sources.values()),
        long_sampling_fps=1,bank='fixed LS4_4 864 visual tokens, recent432',
        training_variants=['content_visible','history_values_blind_same_capacity'],
        head_dimensions=dict(question=2048,visual_mean_max=4096,latent=64,maximum_history_nodes=4),
        training=dict(epochs=20,batch_size=16,seed=20261029,optimizer='AdamW',learning_rate=0.001,
            weight_decay=0.01,gradient_clip=1.0,dropout=0.1,weak_rank_coefficient=0.5,
            long_rank_sample_weight=0.25,route_source_then_class_balanced=True,
            long_rows_never_supervise_route=True,threshold_search=False),
        no_factual_unsupported_class_available=True,UNRESOLVED_not_a_negative_or_recent_label=True,
        current_tasks_64_never_training=True,long_material_previously_opened=True,
        new_unseen_long_generalization_test_ready=False,third_action_not_claimed_trained=True,
        GPU_jobs=0,API_calls=0,downloads=0,pretrained_updates=0,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [data,ROOT/'runs/sember_prepare_20261025_01/selection/requests.json',ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json']}))
    print(json.dumps(read(PREP/'protocol.json'),ensure_ascii=False))
if __name__=='__main__':main()
