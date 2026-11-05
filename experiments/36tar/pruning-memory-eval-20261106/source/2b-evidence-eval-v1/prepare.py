"""Metadata-only selection, never load model scores to choose questions."""
import collections,hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rank(q):return hashlib.sha256(('evidence-eval-v1:'+q['id']).encode()).hexdigest()
def choose(rows,n,cap=2):
    groups=collections.defaultdict(list)
    for q in sorted(rows,key=rank):groups[(q['type'],q['length_bin'])].append(q)
    used=collections.Counter();chosen=[]
    while len(chosen)<n:
        before=len(chosen)
        for key in sorted(groups):
            while groups[key] and used[groups[key][0]['source_id']]>=cap:groups[key].pop(0)
            if groups[key]:
                q=groups[key].pop(0);chosen.append(q);used[q['source_id']]+=1
                if len(chosen)==n:break
        assert len(chosen)>before,'insufficient sources within cap'
    return chosen
def main():
    assert not (P/'panel.json').exists(),'preserve frozen selection'
    old=R/'runs/learned_prune_main_prepare_20261101_01'
    sources={s['video_id']:s for s in read(old/'sources.json')}
    nextqa=[]
    for q in read(old/'eval.json'):
        s=sources[q['video_id']];duration=s['duration_s']
        nextqa.append(dict(q,dataset='NExT-QA',source_id=q['video_id'],duration_s=duration,
            length_bin='10-15s' if duration<15 else '15-20s',query_time_s=duration,
            media_path='datasets/learned-prune-nextqa-v1/'+s['mirror_path'],media_sha256=s['mirror_lfs_sha256']))
    stream=[]
    for q in read(R/'runs/streamingbench_interface_prepare_20261031_01/requests.json'):
        t=q['query_time_s']
        if t is None or t>q['reported_media_duration_s'] or q['modality']!='visual_only':continue
        options=[re.sub(r'^[A-D]\.\s*','',x) for x in q['options']]
        stream.append(dict(q,options=options,dataset='StreamingBench',source_id=q['media_name'],type=q['task_type'],
            duration_s=q['reported_media_duration_s'],length_bin='prefix<180s' if t<180 else 'prefix180-600s' if t<600 else 'prefix>=600s'))
    panel=choose(nextqa,30)+choose(stream,30)
    controls=choose(panel[:30],10)+choose(panel[30:],10)
    profile=choose(panel[:30],12,1) # first interface profile uses already audited short clips
    P.mkdir(parents=True,exist_ok=True)
    labels={q['id']:q['answer_index'] for q in panel if q['dataset']=='NExT-QA'}
    def public(q):return {k:v for k,v in q.items() if k not in ('answer_index','answer_text')}
    for name,data in [('panel.json',[public(q) for q in panel]),('controls.json',[public(q) for q in controls]),
                      ('profile.json',[public(q) for q in profile]),('labels_nextqa.json',labels)]:
        (P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={}
    for dataset in ('NExT-QA','StreamingBench'):
        rows=[q for q in panel if q['dataset']==dataset]
        summary[dataset]=dict(questions=len(rows),sources=len({q['source_id'] for q in rows}),
            types=dict(collections.Counter(q['type'] for q in rows)),length_bins=dict(collections.Counter(q['length_bin'] for q in rows)),
            duration_range_s=[min(q['duration_s'] for q in rows),max(q['duration_s'] for q in rows)],
            query_range_s=[min(q['query_time_s'] for q in rows),max(q['query_time_s'] for q in rows)])
    (P/'selection_summary.json').write_text(json.dumps(dict(panel=summary,profile_questions=12,profile_sources=len({q['source_id'] for q in profile}),
        controls=20,selected_without_predictions=True,streaming_labels_pending=True,actual_stream_input_audit_pending=True),indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
