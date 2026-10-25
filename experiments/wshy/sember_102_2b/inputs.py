"""Immutable official requests and separately locked actual RGB/PTS sampling."""
from pathlib import Path

from ..online_memory_2b.isolation import output_path, write_root
from ..online_memory_2b.protocol import read, safe_path, sha, source_lock as baseline_lock
from ..long_video_2b.protocol import ARMS, validate_queries
from ..long_video_2b.scheduler import replay_source, validate_frames
from types import SimpleNamespace

BUNDLE_REL = 'runs/sember-102-baselines-20261024-01'
MANIFEST_SHA = 'a0d3a2495bfb6531561977a15a69d73855f93a0d491bc3fde3dbb654a74c19e5'
FIXED = dict(sources=18, semantic_questions=102, requests=102, answers=612,
             arms=list(ARMS), max_new_tokens=8, max_generated_tokens=4896,
             reset_checks=90, trained=False, held_out=False)


def bundle_path():
    return output_path(write_root() / BUNDLE_REL)


def source_lock():
    from ..online_memory_2b.frozen import REPO, PACKAGE
    result = baseline_lock()
    for folder in ('long_video_2b', 'long_video_gpu_2b', 'sember_102_2b'):
        for path in (REPO / 'experiments/wshy' / folder).rglob('*'):
            if path.suffix in ('.py', '.sh'):
                result[path.relative_to(REPO).as_posix()] = sha(path)
    path = REPO / 'scripts/run_2b_sember_102.py'
    result[path.relative_to(REPO).as_posix()] = sha(path)
    for path in PACKAGE.rglob('*.py'):
        result[path.relative_to(REPO).as_posix()] = sha(path)
    return dict(sorted(result.items()))


def base_inputs():
    folder = bundle_path()
    if sha(folder / 'sha256.json') != MANIFEST_SHA:
        raise ValueError('official input manifest changed')
    listed = read(folder / 'sha256.json')
    if set(listed) != {'media.json', 'protocol.json', 'requests-no-labels.json', 'labels-offline-only.json'}:
        raise ValueError('official file set changed')
    # Hash label bytes for provenance, without loading labels into model requests.
    for name, digest in listed.items():
        if sha(safe_path(folder, name)) != digest:
            raise ValueError('official input changed: ' + name)
    queries = read(folder / 'requests-no-labels.json')
    groups = validate_queries(queries)
    media = read(folder / 'media.json')
    if (len(queries) != 102 or len(groups) != 18 or len(media) != 18
            or any(not q['id'].startswith('sember:') or q['cohort'] != 'S-EMBER_official_original'
                   or len(q['options']) != 5 for q in queries)):
        raise ValueError('102 unique official original requests / 18 videos required')
    expected_paths = {'data/S-EMBER/' + m['hf_revision'] + '/' + m['relative_path'] for m in media}
    if set(groups) != expected_paths:
        raise ValueError('media/request source binding changed')
    return queries, groups, media


def snapshots(frames, queries):
    result = {}
    packets = (SimpleNamespace(frame_id=f['frame_id'], timestamp=f['pts_s']) for f in frames)
    totals = replay_source(frames, queries, packets, lambda *unused: None,
                           lambda query, prefix: result.update({query['id']: prefix}))
    return result, totals


def counts(sources):
    return dict(sampled_frames=sum(len(item['frames']) for item in sources),
                encoded_pairs=sum(len(item['frames']) // 2 for item in sources),
                final_pending=sum(len(item['frames']) % 2 for item in sources))


def load_bundle(*, verify_media=True):
    _, groups, media = base_inputs()
    folder = bundle_path() / 'sampling'
    listed = read(folder / 'manifest.json')
    expected = {'sources.json', 'snapshots.json'} | {'frames/' + m['video_id'] + '.json' for m in media}
    if set(listed) != expected:
        raise ValueError('sampling file set changed')
    for name, digest in listed.items():
        if sha(safe_path(folder, name)) != digest:
            raise ValueError('sampling file changed: ' + name)
    sources, pinned = read(folder / 'sources.json'), read(folder / 'snapshots.json')
    bypath = {s['source_path']: s for s in sources}
    if len(sources) != 18 or set(bypath) != set(groups) or len(pinned) != 102:
        raise ValueError('sampled source or query count changed')
    result = []
    metadata = {m['video_id']: m for m in media}
    for source in sources:
        path = source['source_path']
        video = safe_path(write_root(), path)
        original = metadata[source['video_id']]
        if (video.stem != source['video_id'] or not video.is_file()
                or video.stat().st_size != original['expected_bytes']
                or source['source_bytes'] != original['expected_bytes']
                or (verify_media and sha(video) != source['source_sha256'])):
            raise ValueError('source media identity changed')
        frames = validate_frames(read(folder / 'frames' / (source['video_id'] + '.json')))
        queries = groups[path]
        if (len(frames) != source['sampled_frames']
                or source['max_query_time_s'] != max(q['query_time_s'] for q in queries)
                or any(f['pts_s'] > source['max_query_time_s'] for f in frames)):
            raise ValueError('sample/query horizon changed')
        expected_prefix, totals = snapshots(frames, queries)
        if any(pinned.get(qid) != prefix for qid, prefix in expected_prefix.items()):
            raise ValueError('pinned causal query prefix changed')
        result.append(dict(source=source, media=video, frames=frames,
                           queries=queries, expected_prefix=expected_prefix))
    return result


def check_prefix(expected, actual):
    if expected != actual or actual['query_padding'] is not False:
        raise ValueError('actual query prefix differs from pinned causal snapshot')
