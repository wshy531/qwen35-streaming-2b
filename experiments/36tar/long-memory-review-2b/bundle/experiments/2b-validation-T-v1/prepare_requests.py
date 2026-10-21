"""Construct answer-blind T inputs after every candidate source is reviewed.

This does not submit a task or load a model. Labels remain local to scoring.
"""
from collections import Counter

from prepare import OUT, ROOT, read, save, sha


def main():
    review = read(OUT/'content_review_receipt.json')
    assert review['questions'] == 119 and review['sources'] == 40
    assert review['model_outputs_used'] is False
    for name, digest in review['source_reviews_sha256'].items():
        assert sha(OUT/'content_reviews'/name) == digest
    all_rows = read(OUT/'content_dispositions119.json')
    eligible = read(OUT/'content_eligible_questions.json')
    traces = {t['source_path']:t for t in read(OUT/'sampling_all40_1fps.json')['videos']}
    requests, scorer = [], []
    for row in eligible:
        original = row['original']
        trace = traces[row['source_path']]
        frames = [dict(frame_id=f['frame_id'],pts_s=f['pts_s'],rgb_sha256=f['rgb_sha256'])
                  for f in trace['frames'] if f['pts_s'] <= original['query_time_s']+1e-9]
        assert frames and len({f['frame_id'] for f in frames}) == len(frames)
        assert all(a['pts_s'] < b['pts_s'] for a,b in zip(frames,frames[1:]))
        requests.append(dict(id=original['id'],cohort='T_new_sources',
            media_path=trace['media_path'],media_sha256=trace['media_sha256'],
            model_visible=dict(question=original['question'],options=original['options'],
                               query_time_s=original['query_time_s'],frames=frames)))
        paired_frames = [frames[i:i+2] if i+1<len(frames) else [frames[i],frames[i]]
                         for i in range(0,len(frames),2)]
        recent8_pts = [f['pts_s'] for pair in paired_frames[-8:] for f in pair]
        recent_begin = min(recent8_pts)
        intervals = original['evidence_intervals']
        coverage = ('all_by_time_span' if all(a >= recent_begin-1e-9 for a,b in intervals)
                    else 'none_by_time_span' if all(b < recent_begin-1e-9 for a,b in intervals)
                    else 'partial_by_time_span')
        scorer.append(dict(row,selection_role=original['selection_role'],
            actual_pairs=len(paired_frames),recent8_begin_s=recent_begin,
            author_evidence_recent8=coverage,coverage_is_proxy_not_necessity=True))
    assert len(requests) == len({r['id'] for r in requests}) == review['eligible_questions']
    assert {r['id'] for r in requests} <= {r['id'] for r in all_rows}
    for r in requests:
        assert set(r) == {'id','cohort','media_path','media_sha256','model_visible'}
        assert set(r['model_visible']) == {'question','options','query_time_s','frames'}
    save(OUT/'requests.json',requests)
    save(OUT/'scorer_rows_LOCAL_ONLY.json',scorer)
    save(OUT/'request_preparation_receipt.json',dict(
        status='answer_blind_inputs_prepared_NOT_runtime_preflight_or_GPU_authorization',
        raw_questions=119,eligible_questions=len(requests),
        disposition_counts=dict(Counter(r['content_verdict'] for r in all_rows)),
        actual_sources=len({r['media_path'] for r in requests}),
        conservative_groups=len({r['source_group_id'] for r in scorer}),
        coverage_proxy_counts=dict(Counter(r['author_evidence_recent8'] for r in scorer)),
        roles=dict(Counter(r['selection_role'] for r in scorer)),
        requests_sha256=sha(OUT/'requests.json'),
        scorer_sha256=sha(OUT/'scorer_rows_LOCAL_ONLY.json'),
        content_review_receipt_sha256=sha(OUT/'content_review_receipt.json'),
        no_answer_or_interval_or_role_fields_in_worker=True,models_loaded=0,GPU_jobs=0))
    print(read(OUT/'request_preparation_receipt.json'))


if __name__ == '__main__':
    main()
