"""Export one dated development annotation packet, never auto-freeze a benchmark."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from audit_development_evidence import source_identity
from audit_sampling import query_snapshot


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--include-gqa',action='store_true',help='Export a new media-complete snapshot, preserve initial packet')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest_path = root/'experiments/2b-memory-eval/data/candidate_manifest.json'
    rows = [r for r in load(manifest_path) if r['split']=='development']
    out = args.out
    media = {r['source_path']:r for r in load(out/'media_inventory.json')['files']}
    if args.include_gqa:
        media.update({r['source_path']:r for r in load(out/'gqa_media_inventory.json')['files']})
    annotations = {r['id']:r for r in load(out/('annotation_review_with_gqa.json'
                   if args.include_gqa else 'annotation_review.json'))['questions']}
    audit = load(out/'audit_final.json')
    audit_by_id = {r['id']:r for r in audit['questions']}
    fields = ['id','dataset','source_path','local_media','media_sha256','task','query_time_s',
        'question','options_json','original_answer_letter','original_answer_text',
        'author_evidence_intervals_json','source_time_audit','current_annotation',
        'quality_flags','parent_identity','parent_status','reviewer_verdict',
        'reviewer_evidence_intervals_json','reviewer_evidence_frames_s_json',
        'visual_only_without_subtitle','negative_full_prefix_reviewed','reviewer_name','reviewer_notes']
    records = []
    for row in rows:
        a = annotations.get(row['id'],{})
        source = audit_by_id[row['id']].get('source_evidence_audit',{})
        canonical,status = source_identity(row['dataset'],row['source_path'])
        intervals = source.get('answer_intervals',row.get('evidence_intervals'))
        flags = []
        state = a.get('readiness','pending_media_and_visual_review')
        for fragment,label in [('future','future_answer'),('burned_in','answer_text_cue'),
                               ('option_overlap','semantic_option_overlap'),
                               ('current_view','current_view_sufficient')]:
            if fragment in state:
                flags.append(label)
        if row['source_path'] not in media:
            flags.append('media_missing')
        if status=='parent_recording_unresolved':
            flags.append('parent_recording_unresolved')
        if row['task']=='HLD':
            flags.append('negative_full_prefix_review_pending')
        records.append(dict(id=row['id'],dataset=row['dataset'],source_path=row['source_path'],
            local_media=media.get(row['source_path'],{}).get('local_path',''),
            media_sha256=media.get(row['source_path'],{}).get('sha256',''),task=row['task'],
            query_time_s=row['query_time_s'],question=row['question'],
            options_json=json.dumps(row['options'],ensure_ascii=False),
            original_answer_letter=chr(65+row['answer_index']),
            original_answer_text=row['options'][row['answer_index']],
            author_evidence_intervals_json=json.dumps(intervals),
            source_time_audit=source.get('status','not_available'),current_annotation=state,
            quality_flags='|'.join(flags),parent_identity=canonical,parent_status=status,
            reviewer_verdict='pending',reviewer_evidence_intervals_json='',
            reviewer_evidence_frames_s_json='',visual_only_without_subtitle='',
            negative_full_prefix_reviewed='',reviewer_name='',reviewer_notes=''))
    table_name='开发集逐题复核表_媒体齐备.csv' if args.include_gqa else '开发集逐题复核表.csv'
    with (out/table_name).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)

    old = load(root/'runs/evidence_review_20261017_01/sampling_05fps.json')
    new = load(out/'sampling_remaining4_05fps.json')
    videos = old['videos']+new['videos']
    if args.include_gqa:
        videos+=load(out/'sampling_gqa12_05fps.json')['videos']
    trace_questions = []
    for row in rows:
        if row['id'] not in annotations:
            continue
        video = next(v for v in videos if v['source_path']==row['source_path'])
        a = annotations[row['id']]
        intervals=a.get('verified_necessary_evidence_intervals') or a.get('author_evidence_intervals')
        snap = query_snapshot(video['frames'],row['query_time_s'],
            intervals,a.get('provisional_inspection_window_s'))
        points = a.get('verified_evidence_frames_s',[])
        snap.update(id=row['id'],annotation_readiness=a['readiness'],
            interval_basis='author_related_segments_not_human_necessity_verification'
                if a.get('author_evidence_intervals') else 'review_supplied_or_unknown',
            reviewed_point_witnesses=[dict(pts_s=t,sampled_exactly=any(abs(f['pts_s']-t)<1e-6
                for f in video['frames'] if f['pts_s']<=row['query_time_s'])) for t in points])
        trace_questions.append(snap)
    combined = dict(status='actual_cpu_sampling_and_storage_indices_not_model_accuracy',
        annotation_metadata_refreshed_without_redecoding=True,videos=videos,questions=trace_questions)
    sampling_name='sampling_all20_05fps.json' if args.include_gqa else 'sampling_all8_05fps.json'
    (out/sampling_name).write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary = dict(status='review_packet_ready_benchmark_not_frozen',development_questions=len(rows),
        development_sources=len({(r['dataset'],r['source_path']) for r in rows}),
        local_media_sources=len(media),local_media_bytes=sum(r['bytes'] for r in media.values()),
        questions_with_media=sum(bool(r['local_media']) for r in records),
        questions_without_media=sum(not r['local_media'] for r in records),
        sampled_real_frames=sum(len(v['frames']) for v in videos),
        decode_seconds=sum(v['elapsed_s'] for v in videos),
        quality_flags=dict(Counter(flag for r in records for flag in r['quality_flags'].split('|') if flag)),
        authoritative_time_audit=audit['action_source_statuses'],
        formal_frozen_questions=0,pretrained_answers_seen=0,gpu_jobs=0,
        original_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    summary_name='review_packet_summary_media_ready.json' if args.include_gqa else 'review_packet_summary.json'
    (out/summary_name).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
