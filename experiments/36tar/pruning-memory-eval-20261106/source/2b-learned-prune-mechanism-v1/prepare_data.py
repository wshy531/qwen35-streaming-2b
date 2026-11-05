"""Freeze a metadata-only, source-disjoint training pilot; never download media."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/learned_prune_data_prepare_20261101_01'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    upstream = OUT / 'upstream'
    rows = list(csv.DictReader((upstream / 'train.csv').open(encoding='utf-8', newline='')))
    mapping = read(upstream / 'map_vid_vidorID.json')
    archive_path = ROOT / 'runs/nextqa_zip_preflight_20261016_01/directory/archive_index.json'
    archive = {r['path']: r for r in read(archive_path)['entries']}
    old_path = ROOT / 'experiments/2b-memory-eval/data/candidate_manifest.json'
    old = read(old_path)
    old_ids = {str(r['source_id']) for r in old if r['dataset'] == 'nextgqa'}
    # The official validation ID set excludes ALL previously used validation cases,
    # including T batches not represented by the oldest candidate manifest.
    val_path = ROOT / 'experiments/2b-memory-eval/data/upstream/nextgqa/val.csv'
    val_ids = {r['video_id'] for r in csv.DictReader(val_path.open(encoding='utf-8'))}
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['video_id']].append(row)
    assert not set(grouped) & (old_ids | val_ids), 'train / previous validation overlap'
    eligible = []
    for vid, qs in grouped.items():
        metadata = {(q['frame_count'], q['width'], q['height']) for q in qs}
        assert len(metadata) == 1
        frame_count = int(qs[0]['frame_count'])
        # Frame count is a workload proxy, NOT an inferred duration or FPS.
        if not 240 <= frame_count <= 480:
            continue
        path = 'NExTVideo/' + mapping[vid] + '.mp4'
        assert path in archive
        rank = hashlib.sha256(('P1-data-v1:' + vid).encode()).hexdigest()
        eligible.append((rank, vid, path))
    eligible.sort()
    assert len(eligible) >= 240
    sources, questions = [], []
    for index, (rank, vid, path) in enumerate(eligible[:240]):
        role = 'train' if index < 160 else ('development' if index < 200 else 'confirmation')
        a = archive[path]
        sources.append(dict(video_id=vid, role=role, official_split='train', mirror_path=path,
                            bytes=a['bytes'], official_zip_crc32=a['crc32'], rank=rank,
                            frame_count=int(grouped[vid][0]['frame_count']), duration_s=None,
                            media_verified=False, parent_capture_id=None))
        for q in sorted(grouped[vid], key=lambda x: int(x['qid'])):
            options = [q[f'a{i}'] for i in range(5)]
            matches = [i for i, v in enumerate(options) if v == q['answer']]
            assert len(matches) == 1, (vid, q['qid'], 'ambiguous original label')
            questions.append(dict(id=f"nextqa-train-{vid}-{q['qid']}", video_id=vid, role=role,
                                  question=q['question'], options=options, answer_index=matches[0],
                                  answer_text=q['answer'], type=q['type'], query_policy='full_clip_EOF',
                                  label_origin='official_NExT_QA_train', sampled_evidence_verified=False))
    assert len({q['id'] for q in questions}) == len(questions)
    assert len({s['video_id'] for s in sources}) == 240
    local_files = list((ROOT / 'datasets/2b-memory-eval').rglob('*.mp4'))
    summary = dict(status='metadata_frozen_media_not_acquired_training_not_started',
                   eligible_videos=len(eligible), selected_videos=240, questions=len(questions),
                   roles={role: dict(videos=sum(s['role'] == role for s in sources),
                                     questions=sum(q['role'] == role for q in questions),
                                     types=dict(collections.Counter(q['type'] for q in questions if q['role'] == role)))
                          for role in ['train', 'development', 'confirmation']},
                   bytes=sum(s['bytes'] for s in sources),
                   previous_validation_overlap=0,
                   local_media_matching_train_ids=len(set(grouped) & {p.stem for p in local_files}),
                   capture_level_independence='unknown; video ID separation only',
                   official_train_sha256=digest(upstream / 'train.csv'),
                   mapping_sha256=digest(upstream / 'map_vid_vidorID.json'),
                   official_archive_sha256=digest(archive_path), previous_val_sha256=digest(val_path),
                   decode_contract=dict(max_duration_s=20, sampling_fps=2, max_real_frames=40,
                                        canvas_hw=[288, 384], truncate=False, question_at='EOF',
                                        reject_policy='retain exclusion log, no replacement by model score',
                                        first_pilot_minimum_admitted=dict(train=100, development=20, confirmation=20)),
                   limitation='Complete temporal extent does not prove sampled pixels preserve every answer. '
                              'No per-question visual or independent label audit has occurred. '
                              'Not a long-memory benchmark; old evaluation sets remain regression only.')
    for name, obj in [('sources.json', sources), ('questions.json', questions), ('data_contract.json', summary)]:
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
