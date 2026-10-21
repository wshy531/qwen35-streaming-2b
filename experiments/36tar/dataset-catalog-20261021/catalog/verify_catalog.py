"""Verify dataset identity/counts with stdlib; optional hashes of existing media only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINS = [('lt30', 0, 30), ('30to60', 30, 60), ('60to180', 60, 180),
        ('180to300', 180, 300), ('300to600', 300, 600), ('ge600', 600, float('inf'))]


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def binof(seconds):
    return next(key for key, lo, hi in BINS if lo <= seconds < hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-root', type=Path, help='Optional project root containing datasets; verify media hashes')
    parser.add_argument('--cluster-project', type=Path, help='Optional project root containing data; verify media hashes')
    args = parser.parse_args()
    manifest = read('MANIFEST.json')['files']
    for rel, expected in manifest.items():
        p = (HERE / rel).resolve()
        assert p.is_relative_to(HERE) and p.is_file()
        assert p.stat().st_size == expected['bytes'] and digest(p) == expected['sha256'], rel
    media, questions, releases, s = [read(n) for n in ('media.json', 'questions.json', 'releases.json', 'summary.json')]
    assert len(media) == len({a['id'] for a in media}) == len({a['sha256'] for a in media}) == 188
    videos = [a for a in media if a['asset_type'] == 'video']
    assert len(videos) == s['media']['videos'] == 182
    assert sum(v['bytes'] for v in videos) == s['media']['bytes']
    assert len(media) - len(videos) == s['static_photos'] == 6
    assert dict(Counter(v['material_kind'] for v in videos)) == {k: v['videos'] for k, v in s['video_kinds'].items()}
    assert dict(Counter(v['duration_bin'] for v in videos)) == s['media']['duration_bins']
    assert all(v['duration_s'] > 0 and binof(v['duration_s']) == v['duration_bin'] for v in videos)
    assert len(questions) == len({q['id'] for q in questions}) == s['all_question_time_items'] == 528
    assert len({q['semantic_question_group'] for q in questions}) == s['semantic_question_groups'] == 526
    assets = {a['id']: a for a in media}
    ids = {q['id']: q for q in questions}
    for q in questions:
        assert q['media_id'] in assets and 0 <= q['answer_index'] < len(q['options'])
        assert q['duration_s'] == assets[q['media_id']]['duration_s']
        assert q['duration_bin'] == binof(q['duration_s'])
    official = [q for q in questions if q['origin'] == 'official']
    assert len(official) == s['official_questions'] == 519
    assert dict(Counter(q['official_task'] for q in official)) == s['official_task_counts']
    assert dict(Counter(q['duration_bin'] for q in official)) == s['official_duration_bins']
    for task, values in s['official_task_by_duration'].items():
        assert dict(Counter(q['duration_bin'] for q in official if q['official_task'] == task)) == values
    initial_union = set()
    for release in releases:
        expected = release['ids']
        assert len(expected) == len(set(expected)) == release['questions']
        assert set(expected) <= ids.keys()
        mids = {ids[uid]['media_id'] for uid in expected}
        assert len(mids) == release['media_files'] and mids == set(release['media_ids'])
        assert dict(Counter(ids[uid]['official_task'] for uid in expected)) == release['task_counts']
        assert all(release['name'] in ids[uid]['memberships'] for uid in expected)
        if release['name'] in {'最初100视频235候选', '补充78题_复用原比较视频', '新增24视频72候选'}:
            assert not initial_union.intersection(expected)
            initial_union.update(expected)
    assert len(initial_union) == 385
    assert len([q for q in official if q['batch'] == 'longstream_official_extra']) == 15
    assert len(read('legacy_checks15.json')) == s['earliest_frozen_natural_checks'] == 15
    planned = read('planned1920_NOT_DOWNLOADED.json')
    assert planned['status'] == 'paused' and len(planned['media']) == 1920
    assert s['planned']['admitted_questions'] == 0 and s['planned']['status'] == 'paused_not_downloaded'
    assert not any(a['local_path'].find('egolife-memory4') >= 0 for a in media)
    verified = 0
    for root, key in ((args.local_root, 'local_path'), (args.cluster_project, 'cluster_path')):
        if root is None:
            continue
        root = root.resolve()
        for a in media:
            path = (root / a[key]).resolve()
            assert path.is_relative_to(root) and path.stat().st_size == a['bytes']
            assert digest(path) == a['sha256'], a[key]
            verified += 1
    print(json.dumps(dict(status='passed', catalog_files=len(manifest), video_files=182, static_photos=6,
        official_questions=519, question_time_items=528, semantic_question_groups=526,
        actual_media_hashes_checked=verified, pretrained_loads=0, GPU_submissions=0), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
