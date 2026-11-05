"""Check metadata and emit media review sheets; does not certify video quality."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def validate(rows):
    seen = set()
    groups = defaultdict(set)
    for row in rows:
        if row['id'] in seen:
            raise ValueError('duplicate question id')
        seen.add(row['id'])
        if row['split'] not in {'development', 'comparison', 'holdout'}:
            raise ValueError('invalid split')
        groups[(row['dataset'], row['source_id'])].add(row['split'])
        source = row.get('source_path')
        if not isinstance(source, str) or not source or '\\' in source or ':' in source:
            raise ValueError('invalid relative media path')
        path = PurePosixPath(source)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('media path escapes dataset root')
        t = row['query_time_s']
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError('invalid query timestamp')
        options = row['options']
        if not options or any(not isinstance(v,str) or not v.strip() for v in options):
            raise ValueError('empty options')
        if len(set(options)) != len(options):
            raise ValueError('duplicate options')
        answer = row['answer_index']
        if type(answer) is not int or not 0 <= answer < len(options):
            raise ValueError('invalid answer index')
        spans = row.get('evidence_intervals')
        if spans is not None:
            if not spans:
                raise ValueError('empty annotated evidence')
            for start,end in spans:
                if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (start,end)):
                    raise ValueError('invalid evidence timestamp')
                if not 0 <= start < end <= t + 1e-9:
                    raise ValueError('evidence outside observed prefix')
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError('same video crosses splits')
    return groups


def bucket(gap):
    return '<=10s' if gap <= 10 else '10-30s' if gap <= 30 else '30-60s' if gap <= 60 else '>60s'


def main():
    path = ROOT/'data/candidate_manifest.json'
    rows = json.loads(path.read_text(encoding='utf-8'))
    groups = validate(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'],row['source_id'])].append(row)
    fields = ['dataset','source_id','split','media_relative_path','questions',
              'tasks','max_query_time_s','media_status','sha256','parent_source_id',
              'cross_dataset_dedup','visual_only','answer_review','evidence_review',
              'sampled_evidence_visible','reviewer','notes']
    with (ROOT/'data/media_review.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for (dataset,source), items in sorted(grouped.items()):
            relative=items[0]['source_path']
            if dataset=='nextgqa':
                relative += '.mp4'
            writer.writerow(dict(dataset=dataset,source_id=source,split=items[0]['split'],
                media_relative_path=dataset+'/'+relative,questions=len(items),
                tasks='|'.join(sorted({r['task'] for r in items})),
                max_query_time_s=max(r['query_time_s'] for r in items),media_status='not_downloaded',
                cross_dataset_dedup='pending',visual_only='pending',answer_review='pending',
                evidence_review='pending',sampled_evidence_visible='pending'))
    gqa=[r for r in rows if r['dataset']=='nextgqa']
    report=dict(status='metadata_checks_passed_media_review_pending',
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sources=len(groups),questions=len(rows),
        nextgqa_gap_since_last_evidence_end=dict(Counter(bucket(r['evidence_gap_s']) for r in gqa)),
        nextgqa_gap_since_first_evidence_start=dict(Counter(bucket(r['query_time_s']-min(a for a,b in r['evidence_intervals'])) for r in gqa)),
        nextgqa_multi_interval_questions=sum(len(r['evidence_intervals'])>1 for r in gqa),
        ovo_missing_evidence_intervals=sum(r['dataset']=='ovo' and r['evidence_intervals'] is None for r in rows),
        invariants=['unique_questions','one_split_per_video_path','bounded_answer_indices',
                    'relative_media_paths','no_evidence_after_query'],
        limitations=['parent_video_and_cross_dataset_overlap_not_verified',
                     'last_evidence_gap_does_not_summarize_all_required_evidence',
                     'no_media_decoding_or_human_review_has_occurred',
                     'holdout_must_not_drive_model_or_prompt_changes'])
    (ROOT/'data/readiness_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
