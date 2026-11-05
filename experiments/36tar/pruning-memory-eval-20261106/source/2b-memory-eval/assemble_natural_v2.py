"""Assemble all149 dispositions and candidate77 from existing reviews, no model."""
from collections import Counter
import csv

from prefix_request import make_request
from prepare_natural_v2 import ROOT,OUT,BASE,read,save,sha
from score_memory_eval import digest


def main():
    original=read(OUT/'original149.json');lock=read(OUT/'scope_lock.json')
    assert sha(OUT/'original149.json')==lock['original149_sha256']
    old37=read(ROOT/'runs/comparison37_review_20261019_01/review.json')['questions']
    old6=read(ROOT/'runs/comparison6_review_20261019_01/review.json')['questions']
    new78=read(OUT/'additional78_review.json')['questions']
    oldoverlay={q['id']:q for q in read(BASE/'content_release_v1/question_annotation_overlay.json')['questions']}
    decisions={}
    for q in old37:
        decisions[q['id']]=dict(stratum=q['stratum'],candidate=q['stratum']!='S',note=q['decision_note'],
            witnesses=q['witnesses'],basis='comparison37_review_20261019_01')
    for q in old6:
        keep=q['id']=='nextgqa:val:8224492254:0'
        decisions[q['id']]=dict(stratum='F' if keep else 'S',candidate=keep,
            note=q['observed']+' 限制：'+q['missing'],witnesses=q['witnesses'],basis='comparison6_review_20261019_01')
    for q in new78:decisions[q['id']]=dict(stratum=q['stratum'],candidate=q['candidate'],note=q['note'],witnesses=q['witnesses'],basis='additional78_review')
    traces={t['source_path']:t for t in read(OUT/'sampling_1fps.json')}
    sheets_old=read(BASE/'gqa_comparison_sheets/review_receipts.json')['sources']
    visible={s['source_path']:{f['frame_id']:f for f in s['frames']} for s in sheets_old}
    for s in read(OUT/'additional78_sheets/receipts.json')['sources']:
        visible.setdefault(s['source_path'],{}).update({f['frame_id']:f for f in s['frames']})
    # Larger-resolution followup was inspected historically for old37 and remains
    # bound by the exact common frame/hash; no failed old item is reintroduced.
    records=[];candidates=[];requests=[]
    for r in original:
        d=decisions.get(r['id'])
        if d is None:
            old=oldoverlay[r['id']]
            d=dict(stratum='S',candidate=False,note='沿原71题已有处置保留：'+old['provisional_role']+'；'+str(old['review'].get('limit','')),
                witnesses=[],basis='unchanged_original71_disposition')
        trace={f['frame_id']:f for f in traces[r['source_path']]['frames']}
        w=[]
        for f in d['witnesses']:
            actual=trace.get(f['frame_id']);assert actual and abs(actual['pts_s']-f['pts_s'])<1e-7 and actual['rgb_sha256']==f['rgb_sha256']
            w.append(dict(frame_id=f['frame_id'],pts_s=f['pts_s'],rgb_sha256=f['rgb_sha256'],source_path=r['source_path'],hash_verified=True))
        row=dict(r,curation={**d,'witnesses':w},formal_evaluation_eligible=False)
        records.append(row)
        if not d['candidate']:continue
        assert w
        candidates.append(row)
        asset=dict(source_path=r['source_path'],sha256=r['media_sha256'],media_path=r['media_path'])
        request=make_request(r,asset,traces[r['source_path']]);request['input_protocol']='natural-v2_1fps_192x256_data_locked_execution_pending'
        assert set(request['model_visible'])=={'question','options','query_time_s','frames','completed_real_pairs','snapshot_tail_pair'}
        requests.append(request)
    assert len(records)==149 and len(candidates)==77
    save(OUT/'all149_dispositions.json',records);save(OUT/'candidate77.json',candidates)
    save(OUT/'model_requests77.json',requests)
    plan=[]
    for source in dict.fromkeys(q['source_path'] for q in candidates):
        ids=sorted({w['frame_id'] for q in candidates if q['source_path']==source for w in q['curation']['witnesses']})
        q=next(q for q in candidates if q['source_path']==source)
        plan.append(dict(source_path=source,media_path=q['media_path'],media_sha256=q['media_sha256'],query_time_s=q['query_time_s'],
            frames=[next(f for f in traces[source]['frames'] if f['frame_id']==i) for i in ids]))
    save(OUT/'model_canvas_review_plan.json',dict(status='candidate_inputs_locked_before_canvas_review',sources=plan,
        candidate77_sha256=sha(OUT/'candidate77.json'),maximum_frames_per_source=32,canvas_hw=[192,256],
        annotation_only=True,old_unresolved_items_reopened=False,max_cpu_seconds=600,max_disk_bytes=16*2**20))
    assert all(len(s['frames'])<=32 for s in plan)
    with (OUT/'全部149题处置.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','source_group_id','question','official_answer','stratum','candidate','note','basis'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for q in records:w.writerow(dict(id=q['id'],source_group_id=q['source_group_id'],question=q['question'],
            official_answer=q['options'][q['answer_index']],**{k:q['curation'][k] for k in fields[4:]}))
    print(dict(original_questions=149,candidates=77,videos=len(plan),strata=dict(Counter(q['curation']['stratum'] for q in candidates)),
        raw_input_trace_sha256=digest([q['model_visible'] for q in requests])))


if __name__=='__main__':main()
