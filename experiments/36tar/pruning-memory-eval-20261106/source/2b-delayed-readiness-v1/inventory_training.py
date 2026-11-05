"""Inventory existing short clips and longer official training candidates."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/delayed_profile_prepare_20261104_01'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    old = 'runs/learned_prune_main_prepare_20261101_01/'
    train = read(old + 'train.json')
    audits = {x['video_id']: x for x in read(
        'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
    eligible = [q for q in train if len(audits[q['video_id']]['samples']) // 2 >= 17]
    previous = read('runs/learned_prune_data_prepare_20261101_01/sources.json')
    excluded = {s['video_id'] for s in previous}
    upstream = ROOT / 'runs/learned_prune_data_prepare_20261101_01/upstream'
    with (upstream / 'train.csv').open(encoding='utf-8', newline='') as handle:
        grouped = defaultdict(list)
        for row in csv.DictReader(handle):
            grouped[row['video_id']].append(row)
    with (ROOT / 'experiments/2b-memory-eval/data/upstream/nextgqa/val.csv').open(
            encoding='utf-8', newline='') as handle:
        validation_ids = {row['video_id'] for row in csv.DictReader(handle)}
    assert not set(grouped) & validation_ids
    mapping = read('runs/learned_prune_data_prepare_20261101_01/upstream/map_vid_vidorID.json')
    archive = {r['path']: r for r in read(
        'runs/nextqa_zip_preflight_20261016_01/directory/archive_index.json')['entries']}
    candidates = []
    for video_id, rows in grouped.items():
        if video_id in excluded:
            continue
        frames = {int(q['frame_count']) for q in rows}
        assert len(frames) == 1
        frame_count = frames.pop()
        if not 481 <= frame_count <= 3600:
            continue
        path = 'NExTVideo/' + mapping[video_id] + '.mp4'
        if path not in archive:
            continue
        entry = archive[path]
        candidates.append(dict(video_id=video_id, archive_path=path,
                               bytes=entry['bytes'], crc32=entry['crc32'],
                               frame_count=frame_count, questions=len(rows),
                               question_types=dict(Counter(q['type'] for q in rows))))
    candidates.sort(key=lambda row: hashlib.sha256(
        ('delayed-training-inventory-v1:' + row['video_id']).encode()).hexdigest())
    report = dict(
        status='inventory_only_not_frozen_training_data',
        existing_train_videos=150, existing_train_questions=len(train),
        existing_17plus_pairs_videos=len({q['video_id'] for q in eligible}),
        existing_17plus_pairs_questions=len(eligible),
        longer_candidate_videos=len(candidates),
        longer_candidate_questions=sum(r['questions'] for r in candidates),
        longer_candidate_bytes=sum(r['bytes'] for r in candidates),
        frame_count_range=[481, 3600], excluded_prior_files=len(excluded),
        duration_verified=False, evidence_intervals_verified=False,
        parent_capture_separation_verified=False,
        remaining=['duration and sampling feasibility', 'evidence intervals in training sources',
                   'delayed query cutoff and later clues', 'source-disjoint split and acquisition budget'],
        candidates=candidates,
    )
    with (OUT / 'training_inventory.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'candidates'}))


if __name__ == '__main__':
    main()
