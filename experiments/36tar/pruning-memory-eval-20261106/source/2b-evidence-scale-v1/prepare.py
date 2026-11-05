"""Expand by metadata only; labels and predictions are bound after freezing IDs."""
import collections, hashlib, json, re
from pathlib import Path
R=Path(__file__).resolve().parents[2]; P=R/'runs/evidence_scale_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def key(q):return hashlib.sha256(('evidence-scale-v1:'+q['id']).encode()).hexdigest()
def extend(pool,initial,n,cap):
    chosen=list(initial);seen={q['id'] for q in chosen};used=collections.Counter(q['source_id'] for q in chosen)
    groups=collections.defaultdict(list)
    for q in sorted(pool,key=key):
        if q['id'] not in seen:groups[q['type'],q['length_bin']].append(q)
    while len(chosen)<n:
        before=len(chosen)
        for k in sorted(groups):
            candidates=[q for q in groups[k] if used[q['source_id']]<cap]
            if candidates:
                # Prefer an additional source over an additional correlated question.
                q=min(candidates,key=lambda q:(used[q['source_id']],key(q)))
                groups[k].remove(q);chosen.append(q);used[q['source_id']]+=1
                if len(chosen)==n:break
        assert len(chosen)>before
    return chosen
def main():
    assert not (P/'panel.json').exists();P.mkdir(exist_ok=True)
    old=read(R/'runs/evidence_eval_prepare_20261102_01/panel_media_bound.json')
    sources={s['video_id']:s for s in read(R/'runs/learned_prune_main_prepare_20261101_01/sources.json')}
    raw=read(R/'runs/learned_prune_main_prepare_20261101_01/eval.json');pools={'NExT-QA':[],'StreamingBench':[]}
    for q in raw:
        s=sources[q['video_id']];d=s['duration_s'];assert s['role']!='train'
        clean={k:v for k,v in q.items() if k not in ('answer_index','answer_text')}
        pools['NExT-QA'].append(dict(clean,dataset='NExT-QA',source_id=q['video_id'],duration_s=d,query_time_s=d,
            length_bin='10-15s' if d<15 else '15-20s',media_path='datasets/learned-prune-nextqa-v1/'+s['mirror_path'],media_sha256=s['mirror_lfs_sha256']))
    media={m['name']:m for m in read(R/'runs/streamingbench_router_prepare_20261101_01/media.json')}
    for q in read(R/'runs/streamingbench_interface_prepare_20261031_01/requests.json'):
        t=q['query_time_s']
        if t is None or t>q['reported_media_duration_s'] or q['modality']!='visual_only':continue
        m=media[q['media_name']]
        pools['StreamingBench'].append(dict(q,options=[re.sub(r'^[A-D]\.\s*','',x) for x in q['options']],dataset='StreamingBench',
            source_id=q['media_name'],type=q['task_type'],duration_s=q['reported_media_duration_s'],
            length_bin='prefix<180s' if t<180 else 'prefix180-600s' if t<600 else 'prefix>=600s',
            media_path='datasets/StreamingBench-eval200-v1/videos/'+q['media_name'],media_sha256=m['sha256']))
    panel=[]
    for ds,cap in [('NExT-QA',3),('StreamingBench',2)]:
        chosen=extend(pools[ds],[q for q in old if q['dataset']==ds],150,cap)
        for q in chosen:
            assert not {'answer_index','answer_text'}.intersection(q)
            q['evidence_protocol']='full_short_clip_2FPS' if ds=='NExT-QA' else 'causal_1FPS_recent8pairs'
            q['evaluation_role']='opened_development_regression'
        panel+=chosen
    assert len({q['id'] for q in panel})==300
    save(P/'panel.json',panel) # IDs frozen before any model output is read.
    labels={q['id']:q['answer_index'] for q in raw}
    sb=read(R/'runs/streamingbench_download_prepare_20261030_01/eval_inputs/labels_LOCAL_ONLY.json')
    labels.update({q['id']:ord(sb[q['id']]['answer'])-65 for q in panel if q['dataset']=='StreamingBench'})
    save(P/'labels_LOCAL_ONLY.json',{q['id']:labels[q['id']] for q in panel})
    reusable=[]
    for folder in ('evidence-nextqa-220067','evidence-stream-merged-20261102-01'):
        for line in (R/'runs'/folder/'answers.jsonl').read_text(encoding='utf-8').splitlines():
            a=json.loads(line)
            if a['protocol']=='B' and a['shift']==0:
                reusable.append(dict(a,reused_from=folder))
    assert len(reusable)==180
    save(P/'reuse.json',reusable)
    regression=[]
    for ds in pools:
        ids=[q['id'] for q in sorted(old,key=key) if q['dataset']==ds][:3]
        regression += [x for x in reusable if x['id'] in ids]
    save(P/'regression.json',regression);assert len(regression)==18
    # 15% rotating evidence panel, no performance-based question selection.
    selected=[]
    for ds,n in [('NExT-QA',23),('StreamingBench',22)]:
        subset=[q for q in panel if q['dataset']==ds]
        selected+=extend(subset,[],n,1)
    save(P/'auxiliary_panel.json',selected)
    audits={r['video_id']:r for r in read(R/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
    summary={}
    for ds in pools:
        qs=[q for q in panel if q['dataset']==ds];vids={q['source_id'] for q in qs}
        pairs=sum((len(audits[v]['samples'])+1)//2 for v in vids) if ds=='NExT-QA' else sum(int(max(q['query_time_s'] for q in qs if q['source_id']==v)//2)+2 for v in vids)
        summary[ds]=dict(questions=len(qs),sources=len(vids),types=dict(collections.Counter(q['type'] for q in qs)),
            length_bins=dict(collections.Counter(q['length_bin'] for q in qs)),duration_range_s=[min(q['duration_s'] for q in qs),max(q['duration_s'] for q in qs)],
            query_range_s=[min(q['query_time_s'] for q in qs),max(q['query_time_s'] for q in qs)],
            answers=750,reused=90,new_answers=660,endpoint_regressions=9,warmups=5,vision_call_upper_bound=4*pairs+4)
    save(P/'selection.json',dict(questions=300,new_questions=240,reused_answers=180,new_answers=1320,regression_answers=18,warmups=10,
        max_new_generated_tokens=(1320+18+10)*128,arms=['full','fixed12','heuristic11','mask_ce','text_only'],datasets=summary,
        source_split_claim='opened development/regression; no unseen-test claim',selection_without_predictions=True,auxiliary_questions=45,API_calls=0,training=0))
    print(json.dumps(read(P/'selection.json'),ensure_ascii=False))
if __name__=='__main__':main()
