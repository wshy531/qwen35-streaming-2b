"""Prepare metadata candidates only; no claim of media or annotation review."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = '2b-memory-v1-20261016'

def rank(value):
    return hashlib.sha256((SEED + str(value)).encode()).hexdigest()

def save(name, value):
    (ROOT / 'data' / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    upstream = ROOT / 'data/upstream'
    rows = list(csv.DictReader((upstream / 'nextgqa/val.csv').open(encoding='utf-8')))
    grounding = json.loads((upstream / 'nextgqa/gsub_val.json').read_text(encoding='utf-8'))
    mapping = json.loads((upstream / 'nextgqa/map_vid_vidorID.json').read_text(encoding='utf-8'))
    ovo = json.loads((upstream / 'ovo/ovo_bench_new.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    excluded = Counter()
    for row in rows:
        vid = row['video_id']
        info = grounding.get(vid, {})
        spans = info.get('location', {}).get(row['qid'], [])
        duration = info.get('duration', 0)
        options = [row[f'a{i}'] for i in range(5)]
        if not spans or not duration or any(not (0 <= a < b <= duration + .1) for a,b in spans):
            excluded['nextgqa_invalid_or_missing_grounding'] += 1
            continue
        if options.count(row['answer']) != 1:
            excluded['nextgqa_ambiguous_answer'] += 1
            continue
        groups[('nextgqa', vid)].append(dict(
            id=f"nextgqa:val:{vid}:{row['qid']}", dataset='nextgqa', source_id=vid,
            source_path=mapping.get(vid), task=row['type'], question=row['question'],
            options=options, answer_index=options.index(row['answer']),
            evidence_intervals=spans, query_time_s=duration,
            query_policy='original_full_clip_end', duration_s=duration,
            evidence_gap_s=max(0, duration-max(b for a,b in spans))))
    for row in ovo:
        if row['task'] not in {'EPM', 'ASI', 'HLD'}:
            continue
        if not isinstance(row.get('options'), list) or not isinstance(row.get('gt'), int):
            excluded['ovo_invalid_options'] += 1
            continue
        groups[('ovo', row['video'])].append(dict(
            id=f"ovo:{row['id']}", dataset='ovo', source_id=row['video'],
            source_path=row['video'], task=row['task'], question=row['question'],
            options=row['options'], answer_index=row['gt'],
            query_time_s=row['realtime'], query_policy='official_query_time',
            evidence_intervals=None, evidence_gap_s=None))
    selected = []
    source_counts = {}
    for dataset, limits in [('nextgqa', (12,24,24)), ('ovo', (8,16,16))]:
        keys = sorted((k for k in groups if k[0] == dataset), key=lambda k: rank(k[1]))
        source_counts[dataset] = len(keys)
        for i,key in enumerate(keys[:sum(limits)]):
            split = 'development' if i < limits[0] else 'comparison' if i < sum(limits[:2]) else 'holdout'
            candidates = sorted(groups[key], key=lambda r: rank(r['id']))
            # Round robin by task limits a video's repetitive question templates.
            chosen = []
            for task in sorted({r['task'] for r in candidates}):
                chosen.append(next(r for r in candidates if r['task'] == task))
            chosen = (chosen + [r for r in candidates if r not in chosen])[:3]
            for row in chosen:
                row.update(split=split, review_status='pending_media_and_human_review',
                           source_group_status='pending_parent_video_and_cross_dataset_dedup')
                selected.append(row)
    ids = [r['id'] for r in selected]
    assert len(ids) == len(set(ids))
    source_splits = defaultdict(set)
    for row in selected:
        assert 0 <= row['answer_index'] < len(row['options'])
        source_splits[(row['dataset'],row['source_id'])].add(row['split'])
    assert all(len(s) == 1 for s in source_splits.values())
    save('candidate_manifest.json', selected)
    summary = dict(status='metadata_candidates_not_frozen_evaluation_set', seed=SEED,
        upstream_nextgqa_rows=len(rows), upstream_ovo_rows=len(ovo),
        upstream_ovo_tasks=dict(Counter(r['task'] for r in ovo)),
        eligible_sources=source_counts, selected_sources=len(source_splits),
        selected_questions=len(selected), exclusions=dict(excluded),
        tasks=dict(Counter(r['dataset']+':'+r['task'] for r in selected)),
        splits={s:dict(sources=len({(r['dataset'],r['source_id']) for r in selected if r['split']==s}),
                       questions=sum(r['split']==s for r in selected))
                for s in ['development','comparison','holdout']},
        upstream_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(upstream.rglob('*')) if p.is_file()})
    save('audit_summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='upstream_sha256'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
