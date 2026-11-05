"""Scorer-side memory-pressure strata, no labels to policy or inference.

Absent author interval coverage is a proxy, never proof that old memory is
necessary: annotations can be nonexclusive and answer cues can recur later.
"""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    out=ROOT/'runs/memory_pressure_strata_20261019_01';out.mkdir(exist_ok=False)
    base=ROOT/'runs/dataset_build_20261018_01/content_release_v1'
    questions=read(base/'question_annotation_overlay.json')['questions']
    traces={v['source_path']:v for v in read(base/'actual_sampling_traces.json')['videos']}
    rows=[]
    for q in questions:
        r=q['original'];query=r['query_time_s']
        frames=[f for f in traces[r['source_path']]['frames'] if f['pts_s']<=query+1e-9]
        pairs=(len(frames)+1)//2;recent=frames[2*max(0,pairs-8):]
        intervals=r.get('evidence_intervals')
        hits=lambda fs:[any(a-1e-9<=f['pts_s']<=b+1e-9 for f in fs) for a,b in intervals] if intervals else None
        full_hit,recent_hit=hits(frames),hits(recent)
        if pairs<=8:proxy='no_8pair_storage_pressure'
        elif not intervals:proxy='author_intervals_unavailable'
        elif not all(full_hit):proxy='full_input_misses_some_author_intervals'
        elif not any(recent_hit):proxy='recent_misses_all_author_intervals'
        elif not all(recent_hit):proxy='recent_misses_some_author_intervals'
        else:proxy='recent_covers_all_author_intervals'
        rows.append(dict(id=q['id'],split=r['split'],dataset=r['dataset'],source_path=r['source_path'],
            source_group_id=q['source_group_id'],parent_identity_status=q['parent_identity_status'],
            content_domain=q['content_domain'],question=r['question'],task=r['task'],
            query_time_s=query,full_real_frames=len(frames),full_pairs=pairs,
            recent_real_frames=len(recent),recent_pairs=min(8,pairs),
            first_recent_pts_s=recent[0]['pts_s'] if recent else None,
            review_verdict=q['review']['verdict'],existing_provisional_role=q['provisional_role'],
            author_intervals=intervals,full_author_interval_hits=full_hit,recent_author_interval_hits=recent_hit,
            memory_pressure_proxy=proxy,
            requires_old_evidence_proven=False,formal_eligibility_changed=False))
    assert len(rows)==235 and len({r['id'] for r in rows})==235
    summaries={}
    for split in ['development','comparison','holdout']:
        subset=[r for r in rows if r['split']==split]
        summaries[split]=dict(questions=len(subset),strata=dict(Counter(r['memory_pressure_proxy'] for r in subset)),
            nextgqa_strata=dict(Counter(r['memory_pressure_proxy'] for r in subset if r['dataset']=='nextgqa')),
            naturalistic_provisional_recent_absent=[r['id'] for r in subset
                if r['existing_provisional_role']=='provisional_naturalistic_diagnostic' and r['memory_pressure_proxy']=='recent_misses_all_author_intervals'])
    payload=dict(status='fixed_metadata_stratification_not_question_selection_or_ability_score',
        rows=rows,splits=summaries,model_outputs_used=False,author_answers_used=False,
        heldout_images_reviewed=False,original_splits_or_rows_changed=False,
        original_manifest_sha256=hashlib.sha256((ROOT/'experiments/2b-memory-eval/data/candidate_manifest.json').read_bytes()).hexdigest(),
        limitations=['No-pressure examples remain as controls; do not delete to inflate gains.',
            'Author interval misses do not prove information absent from all recent frames.',
            'Use original full comparison list and predeclared strata; do not pick cases from model success.'])
    (out/'strata.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['id','split','dataset','source_path','question','full_pairs','recent_pairs','first_recent_pts_s',
        'memory_pressure_proxy','review_verdict','existing_provisional_role']
    with (out/'235题记忆压力分层.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in fields})
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':main()
