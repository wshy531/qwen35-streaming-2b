"""Bind existing frozen heads and all scheduled tasks, no model-based selection."""
import math
from collections import Counter
from selector_qa_common import ROOT,PREP,TRAIN,HEADS,ARMS,read,save,sha

def main():
    if PREP.exists():raise FileExistsError('Preserve fixed preparation')
    old=ROOT/'runs/evidence_selection_prepare_20261029_01'
    plan=read(old/'runtime_plan.json');tasks=read(old/'inputs/requests.json')
    labels=read(old/'audit/labels_LOCAL_ONLY.json')
    for t in tasks:t['evaluation_group']='regression64'
    profiles={r['task_id']:r for r in plan['tasks']}
    # Stronger completed references include actual IDs/positions for all 640 rows.
    reference_rows=[__import__('json').loads(s) for s in (ROOT/'runs/2b-evidence-selection-219515/answers.jsonl').read_text(encoding='utf-8').splitlines()]
    refs=[{k:r[k] for k in ('task_id','arm','shift','input_ids_sha256','position_ids_sha256','generated_ids','text')} for r in reference_rows if r['arm'] in ('recent','full')]
    assert len(refs)==640
    sources=[dict(s,relative_path='datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos/'+s['name'],evaluation_group='regression64') for s in plan['sources']]
    checkpoints=[dict(task_id=r['task_id'],committed_pairs=r['committed_pairs'],cutoff_s=r['cutoff_s'],nodes=r['nodes']) for r in plan['tasks']]
    targets=read(TRAIN/'long_supervision_LOCAL_ONLY.json')
    wanted=[t for t in read(TRAIN/'long_requests.json') if targets[t['id']]['role']=='heldout_development']
    full={r['question_id']:r for r in read(ROOT/'runs/sember_prepare_20261025_01/selection/requests.json')}
    long_labels={r['question_id']:r for r in read(ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json')}
    nodes={r['id']:r for r in read(ROOT/'runs/selector-features-219523/nodes.json')}
    for t in wanted:
        qid=t['id'][5:];r=full[qid];tr=nodes[t['id']]
        tasks.append(dict(id=t['id'],question=t['question'],options=r['options'],source_name=r['video_id']+'.mp4',
            query_time_s=t['query_time_s'],category=r['question_category'],role='official_reference',
            evaluation_group='long36_opened_source_heldout'))
        labels[t['id']]=long_labels[qid]['correct_index']
        pairs=(math.floor(t['query_time_s'])+1)//2
        recent=[dict(first=i,last=i,begin=2.*i,end=2.*i+1,center=2.*i+.5,tokens=108,exact_pair=True) for i in range(pairs-4,pairs)]
        checkpoints.append(dict(task_id=t['id'],committed_pairs=pairs,cutoff_s=2.*pairs-1,nodes=tr['history']+recent))
    for s in read(TRAIN/'long_sources.json'):
        relevant=[t for t in wanted if t['source_id']==s['video_id']]
        if not relevant:continue
        maximum=max((math.floor(t['query_time_s'])+1)//2 for t in relevant)
        sources.append(dict(name=s['video_id']+'.mp4',bytes=s['bytes'],sha256=s['sha256'],
            relative_path='datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/'+s['relative_path'],
            max_pairs=maximum,cutoff_s=2.*maximum-1,check_frames=[],evaluation_group='long36_opened_source_heldout'))
    assert len(tasks)==100 and len(sources)==24 and len({s['name'] for s in sources})==24
    assert len({t['id'] for t in tasks})==100 and sum(t['evaluation_group']=='regression64' for t in tasks)==64
    for name in ('content_visible','history_blind'):
        trained=__import__('torch').load(HEADS/(name+'.pt'),weights_only=True,map_location='cpu')['train_ids']
        assert not set(trained)&{t['id'] for t in tasks}
    vision=sum(s['max_pairs'] for s in sources)
    protocol=dict(tasks=100,source_files=24,groups=dict(Counter(t['evaluation_group'] for t in tasks)),
        arms=ARMS,option_shifts=list(range(5)),formal_answers=3000,warmup_answers=6,
        maximum_generation_tokens=3006*8,selector_question_prefills=1500,warmup_question_prefills=3,
        visual_forwards=vision,pretrained_loads=1,CPU_threads=4,RAM_GiB=24,GPU_count=1,
        wall_minutes=60,log_MiB=64,result_MiB=96,no_requeue=True,automatic_retry=False,
        max_new_tokens=8,canvas=[288,384],fps=1,bank_visual_tokens=864,recent_visual_tokens=432,
        selector_history_tokens_max=216,fresh_selector_question_encoding_each_answer=True,
        bank_and_vision_shared_across_methods=True,standalone_end_to_end_cost=False,
        matched_latest_uses_visible_route_and_count=True,reference_rows=640,
        one_fixed_evaluation_no_retraining_or_threshold_search=True,
        quality_goal='Report current/old QA changes, corrections and losses; quality-preserving lower cost is useful even without net old-QA gain',
        no_automatic_pass_fail_quality_gate=True,new_test_sources_claimed=False,
        approved_GPU_submission=False,training_steps=0,API_calls=0,downloads=0)
    save(PREP/'requests.json',tasks);save(PREP/'labels_LOCAL_ONLY.json',labels)
    save(PREP/'plan.json',dict(sources=sources,checkpoints=checkpoints))
    save(PREP/'references_LOCAL_ONLY.json',refs);save(PREP/'protocol.json',protocol)
    save(PREP/'head_lock.json',{name:sha(HEADS/(name+'.pt')) for name in ('content_visible','history_blind')})
    print(protocol)

if __name__=='__main__':main()
