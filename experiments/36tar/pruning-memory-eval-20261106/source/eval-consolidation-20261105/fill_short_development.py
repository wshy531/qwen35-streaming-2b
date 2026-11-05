"""Freeze and verify a small source-disjoint NExT-QA development supplement."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote
import zlib

import av
import httpx

from build import load, rank, ROOT


OUT = ROOT / 'runs/eval_consolidation_20261105_02'
PRIOR = ROOT / 'runs/eval_consolidation_20261105_01'
MEDIA = ROOT / 'datasets/eval-consolidation-short-dev-v1'
UPSTREAM = 'runs/learned_prune_data_prepare_20261101_01/upstream'
REPO = 'jinyoungkim/NExT-GQA'
REVISION = '5d1c262802efe7e5331629ca01e67916867590a5'
CAP = 50 * 2**20
FILE_LIMIT = 8


def write(name, data):
    with (OUT / f'{name}.json').open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def prepare():
    queue = load('runs/eval_consolidation_20261105_01/new_short_probe_queue.json')
    upstream = load('runs/nextqa_zip_preflight_20261016_01/directory/archive_index.json')
    archives = {PurePosixPath(row['path']).stem: row for row in upstream['entries']
                if row['path'].endswith('.mp4')}
    source_map = load(f'{UPSTREAM}/map_vid_vidorID.json')
    used = set()
    for manifest in sorted((ROOT / 'datasets/2b-memory-eval').glob('nextgqa*/download_manifest.json')):
        for row in load(manifest.relative_to(ROOT).as_posix())['files']:
            name = str(row.get('source_path', ''))
            used.add(PurePosixPath(name).stem)
    known_groups = {source_map[v] for v in used if v in source_map}
    candidates = [q for q in queue if q['probe_band_at_assumed_30fps'] == '40-60'
                  and q['video_id'] in archives and q['video_id'] not in used
                  and q['source_group'] not in known_groups]
    chosen = sorted(candidates, key=lambda q: (abs(q['frame_count'] - 1500), rank(q['video_id'])))[:FILE_LIMIT]
    files = [{**q, 'archive_path': archives[q['video_id']]['path'],
              'bytes': archives[q['video_id']]['bytes'],
              'crc32': archives[q['video_id']]['crc32']} for q in chosen]
    assert len(files) == FILE_LIMIT and sum(q['bytes'] for q in files) <= CAP
    write('short_acquisition_plan', dict(repo=REPO, revision=REVISION,
          media_root=MEDIA.relative_to(ROOT).as_posix(), file_count=len(files),
          total_bytes=sum(q['bytes'] for q in files), cap_bytes=CAP, files=files,
          allocation='development_only', selection='nearest 1500 official frames, then fixed hash'))
    print(json.dumps(dict(files=len(files), bytes=sum(q['bytes'] for q in files))))


def fetch():
    plan = load('runs/eval_consolidation_20261105_02/short_acquisition_plan.json')
    files = plan['files']
    assert plan['repo'] == REPO and plan['revision'] == REVISION
    assert len(files) == FILE_LIMIT and sum(r['bytes'] for r in files) <= CAP
    write('acquisition_started', dict(files=len(files), max_bytes=CAP, retries=0))
    MEDIA.mkdir(exist_ok=False)
    started = time.monotonic()
    complete = []
    written = 0
    try:
        with httpx.Client(timeout=httpx.Timeout(25, connect=10), follow_redirects=True) as client:
            with client.stream('POST', f'https://huggingface.co/api/datasets/{REPO}/paths-info/{REVISION}',
                               json={'paths': [r['archive_path'] for r in files], 'expand': False}) as response:
                response.raise_for_status()
                body = bytearray()
                for block in response.iter_bytes(8192):
                    body.extend(block)
                    assert len(body) <= 131072
            remote = {r['path']: r for r in json.loads(body)}
            for row in files:
                archive_path = row['archive_path']
                info = remote[archive_path]
                assert info['size'] == row['bytes']
                expected_sha = info['lfs']['oid']
                assert re.fullmatch('[a-f0-9]{64}', expected_sha)
                assert PurePosixPath(archive_path).name == row['video_id'] + '.mp4'
                final = MEDIA / PurePosixPath(archive_path).name
                partial = final.with_suffix('.partial')
                sha, crc, length = hashlib.sha256(), 0, 0
                url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(archive_path)}'
                with client.stream('GET', url) as response, partial.open('xb') as output:
                    response.raise_for_status()
                    for block in response.iter_bytes(65536):
                        length += len(block)
                        written += len(block)
                        assert length <= row['bytes'] and written <= CAP
                        if time.monotonic() - started > 240:
                            raise TimeoutError('download exceeded 240 seconds')
                        sha.update(block)
                        crc = zlib.crc32(block, crc)
                        output.write(block)
                assert length == row['bytes'] and sha.hexdigest() == expected_sha
                assert f'{crc:08x}' == row['crc32']
                partial.rename(final)
                complete.append({**row, 'media_sha256': expected_sha,
                                 'media_path': final.relative_to(ROOT).as_posix()})
    except Exception as error:
        write('acquisition_stopped', dict(error_type=type(error).__name__,
              complete=complete, transferred_bytes=written))
        raise RuntimeError('Bounded acquisition stopped; see local receipt') from None
    write('acquisition_complete', dict(files=complete, transferred_bytes=written))
    print(json.dumps(dict(files=len(complete), bytes=written)))


def admit():
    receipt = load('runs/eval_consolidation_20261105_02/acquisition_complete.json')
    old = load('runs/eval_consolidation_20261105_01/development.json')
    regression = load('runs/eval_consolidation_20261105_01/regression_ids.json')
    labels = load('runs/eval_consolidation_20261105_01/development_labels_LOCAL_ONLY.json')
    with (ROOT / UPSTREAM / 'train.csv').open(encoding='utf-8-sig', newline='') as stream:
        questions = defaultdict(list)
        for q in csv.DictReader(stream):
            questions[q['video_id']].append(q)
    used_groups = {r.get('source_group') for r in old}
    used_hashes = {r.get('media_sha256') for r in old}
    order = load('runs/training_order_main_prepare_20261104_02/panel_LOCAL_ONLY.json')
    source_map = load(f'{UPSTREAM}/map_vid_vidorID.json')
    assert all(v in source_map for v in order['train'])
    blocked_groups = used_groups | {source_map[v] for v in order['train']}
    blocked_hashes = used_hashes | {order['groups'][v]['sha256'] for v in order['train']}
    audit = []
    for row in receipt['files']:
        media = ROOT / row['media_path']
        assert hashlib.sha256(media.read_bytes()).hexdigest() == row['media_sha256']
        with av.open(str(media)) as container:
            video = container.streams.video[0]
            video.thread_count = 1
            duration = float(video.duration * video.time_base) if video.duration is not None else container.duration / 1e6
            count, previous_time = 0, -float('inf')
            for frame in container.decode(video):
                assert frame.time is not None and frame.time > previous_time
                previous_time = frame.time
                count += 1
        eligible = (40 <= duration < 60 and count == row['frame_count']
                    and row['source_group'] not in blocked_groups
                    and row['media_sha256'] not in blocked_hashes)
        audit.append(dict(row, duration_s=duration, decoded_frames=count, eligible=eligible))
    write('short_media_audit', audit)
    candidates = [r for r in audit if r['eligible']]
    assert len({r['source_group'] for r in candidates}) == len(candidates)
    assert len({r['media_sha256'] for r in candidates}) == len(candidates)
    added = []
    types = Counter(q['type'][0] for q in old if q.get('length_bin') == '40-60')
    for _ in range(2):
        for row in candidates:
            if len(added) == 7:
                break
            vid = row['video_id']
            valid = []
            for q in questions[vid]:
                qid = f'nextqa-train-{vid}-{q["qid"]}'
                options = [q[f'a{i}'] for i in range(5)]
                if qid not in labels and q['answer'] in options and len(set(options)) == 5:
                    valid.append((q, qid, options))
            if not valid:
                continue
            q, qid, options = min(valid, key=lambda v: (types[v[0]['type'][0]], rank(v[1])))
            labels[qid] = options.index(q['answer'])
            types[q['type'][0]] += 1
            added.append(dict(id=qid, dataset='NExT-QA', source_id=vid,
                              source_group=row['source_group'], type=q['type'],
                              duration_s=row['duration_s'], query_time_s=row['duration_s'],
                              length_bin='40-60', query_policy='full_clip_EOF',
                              question=q['question'], options=options,
                              media_path=row['media_path'], media_sha256=row['media_sha256']))
    combined = old + added
    ids = {q['id'] for q in combined}
    regression += [q['id'] for q in added]
    assert len(ids) == len(combined) and ids == set(labels)
    assert len(regression) == len(set(regression)) and set(regression) <= ids
    assert all(0 <= labels[q['id']] < len(q['options']) for q in combined)
    bands = Counter(q['length_bin'] for q in combined if q['dataset'] == 'NExT-QA')
    assert all(n <= 15 for n in bands.values())
    write('development', combined)
    write('regression_ids', regression)
    write('development_labels_LOCAL_ONLY', labels)
    write('short_development_added', added)
    write('excluded_from_future_holdout', dict(video_ids=[q['video_id'] for q in audit],
          source_groups=[q['source_group'] for q in audit], sha256s=[q['media_sha256'] for q in audit]))
    summary = dict(development=len(combined), regression=len(regression), added=len(added),
                   eligible_sources=len(candidates), short_bands=dict(bands),
                   short_source_count=len({q['source_group'] for q in combined if q['dataset'] == 'NExT-QA'}),
                   validation_admitted=0)
    write('summary', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'fetch', 'admit'))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    {'prepare': prepare, 'fetch': fetch, 'admit': admit}[args.stage]()
