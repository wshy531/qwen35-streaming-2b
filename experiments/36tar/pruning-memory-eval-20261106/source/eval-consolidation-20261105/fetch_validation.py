"""Fetch the approved validation pilot using one request stream at a time."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
from urllib.parse import quote
import zlib

import httpx

from build import ROOT

sys.path.insert(0, str(ROOT / 'experiments/streamingbench-download-v1'))
from download_members import Inflate, local_header

PLAN = ROOT / 'runs/eval_consolidation_20261105_02'
OUT = ROOT / 'runs/eval_consolidation_20261105_03'
STREAM = 'datasets/StreamingBench-validation-v1'
SHORT = 'datasets/eval-consolidation-short-validation-v1'
BYTE_CAP = int(1.6 * 2**30)
TIME_CAP = 3600


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def short_metadata(client, rows, repo, revision, chunks):
    expected = {r['official_zip']['path']: r for r in rows}
    url = f'https://huggingface.co/api/datasets/{repo}/paths-info/{revision}'
    with client.stream('POST', url, json={'paths': list(expected), 'expand': False}) as response:
        response.raise_for_status()
        body = bytearray()
        for block in chunks(response):
            body.extend(block)
            if len(body) > 2**20:
                raise ValueError('Metadata limit exceeded')
    found = {row['path']: row for row in json.loads(body)}
    if found.keys() != expected.keys():
        raise ValueError('Unexpected metadata file set')
    for name, row in found.items():
        if row['size'] != expected[name]['official_zip']['bytes'] or not re.fullmatch('[0-9a-f]{64}', row['lfs']['oid']):
            raise ValueError('Metadata size or SHA mismatch')
    return found


def get_short(client, row, metadata, base, destination, chunks):
    archive = row['official_zip']
    path = archive['path']
    info = metadata[path]
    sha, crc, length = hashlib.sha256(), 0, 0
    part = destination.with_suffix('.partial')
    with client.stream('GET', base + quote(path)) as response, part.open('xb') as output:
        response.raise_for_status()
        for block in chunks(response):
            length += len(block)
            if length > archive['bytes']:
                raise ValueError('Short media size exceeded')
            sha.update(block)
            crc = zlib.crc32(block, crc)
            output.write(block)
    if (length, f'{crc:08x}', sha.hexdigest()) != (archive['bytes'], archive['crc32'], info['lfs']['oid']):
        raise ValueError('Short media size, CRC or SHA mismatch')
    part.rename(destination)
    return sha.hexdigest()


def get_stream(client, row, base, destination, chunks):
    member = row['member']

    def require_range(response, first, last):
        response.raise_for_status()
        expected = f"bytes {first}-{last}/{row['archive_bytes']}"
        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
            raise ValueError('Exact byte range required')

    def url(first, last):
        return base + quote(row['archive'], safe='') + f'?download=true&validation_range={first}-{last}'

    def headers(first, last):
        return {'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity'}

    offset = member['local_header_offset']
    end = offset + 1023
    with client.stream('GET', url(offset, end), headers=headers(offset, end)) as response:
        require_range(response, offset, end)
        head = bytearray()
        for block in chunks(response):
            head.extend(block)
            if len(head) > 1024:
                raise ValueError('ZIP header size exceeded')
        if len(head) != 1024:
            raise ValueError('Incomplete ZIP header')
    first = offset + local_header(head, member)
    last = first + member['compressed_bytes'] - 1
    if not 0 <= first <= last < row['archive_bytes']:
        raise ValueError('ZIP member range outside archive')
    part = destination.with_suffix('.partial')
    with client.stream('GET', url(first, last), headers=headers(first, last)) as response, part.open('xb') as output:
        require_range(response, first, last)
        decoder = Inflate(output, member)
        received = 0
        for block in chunks(response):
            received += len(block)
            if received > member['compressed_bytes']:
                raise ValueError('ZIP member size exceeded')
            decoder.feed(block)
        if received != member['compressed_bytes']:
            raise ValueError('Incomplete ZIP member')
        sha = decoder.finish()
    part.rename(destination)
    return sha


def main():
    plan = read(PLAN / 'stream_validation_acquisition_plan.json')
    short = read(PLAN / 'short_validation_probe_queue.json')
    short_plan = read(PLAN / 'short_acquisition_plan.json')
    main_rows, diagnosis = plan['pilot']['main'], plan['pilot']['diagnosis']
    assert (len(main_rows), len(diagnosis), len(short)) == (12, 4, 112)
    streams = main_rows + diagnosis
    expected_bytes = sum(r['member']['compressed_bytes'] + 1024 for r in streams)
    expected_bytes += sum(r['official_zip']['bytes'] for r in short)
    assert expected_bytes == 1567063904 and expected_bytes < BYTE_CAP
    assert len({r['name'] for r in streams}) == 16
    assert len({r['video_id'] for r in short}) == 112
    assert all(re.fullmatch(r'sample_\d+_real\.mp4', r['name']) for r in streams)
    assert all(re.fullmatch(r'\d+', r['video_id']) for r in short)
    for relative in (STREAM, SHORT):
        assert not (ROOT / relative).exists()
    OUT.mkdir(exist_ok=False)
    for relative in (STREAM, SHORT):
        (ROOT / relative).mkdir()
    started = time.monotonic()
    state = dict(status='running', byte_cap=BYTE_CAP, seconds_cap=TIME_CAP, attempts_per_file=1,
                 expected_bytes=expected_bytes, body_bytes=0, complete=[], failed=[])

    def checkpoint():
        state['seconds'] = round(time.monotonic() - started, 2)
        save(OUT / 'acquisition.json', state)

    def budget():
        if time.monotonic() - started >= TIME_CAP or state['body_bytes'] >= BYTE_CAP:
            raise TimeoutError('Batch budget exhausted')

    def chunks(response):
        for block in response.iter_bytes(262144):
            state['body_bytes'] += len(block)
            budget()
            yield block

    stream_base = f"https://huggingface.co/datasets/{plan['dataset_repo']}/resolve/{plan['revision']}/"
    short_repo, short_rev = short_plan['repo'], short_plan['revision']
    short_base = f'https://huggingface.co/datasets/{short_repo}/resolve/{short_rev}/'
    checkpoint()
    try:
        with httpx.Client(timeout=httpx.Timeout(45, connect=12), follow_redirects=True) as client:
            try:
                metadata = short_metadata(client, short, short_repo, short_rev, chunks)
            except (httpx.HTTPError, ValueError) as error:
                state['failed'].append(dict(panel='short', stage='metadata',
                                            error=type(error).__name__))
                metadata = None
            panels = [('short', short if metadata else [], SHORT), ('main', main_rows, STREAM),
                      ('diagnosis', diagnosis, STREAM)]
            for panel, rows, target in panels:
                consecutive_failures = 0
                for row in rows:
                    budget()
                    name = row['video_id'] + '.mp4' if panel == 'short' else row['name']
                    destination = ROOT / target / name
                    state['current'] = dict(panel=panel, name=name)
                    checkpoint()
                    try:
                        if panel == 'short':
                            sha = get_short(client, row, metadata, short_base, destination, chunks)
                        else:
                            sha = get_stream(client, row, stream_base, destination, chunks)
                        state['complete'].append(dict(panel=panel, name=name, sha256=sha,
                            bytes=destination.stat().st_size, media_path=destination.relative_to(ROOT).as_posix()))
                        consecutive_failures = 0
                    except (httpx.HTTPError, ValueError, zlib.error) as error:
                        code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                        state['failed'].append(dict(panel=panel, name=name, error=type(error).__name__, code=code))
                        consecutive_failures += 1
                        if code in (401, 403, 429) or consecutive_failures >= 3:
                            break
                    finally:
                        checkpoint()
                    if len(state['complete']) % 8 == 0:
                        print(f"{len(state['complete'])}/128 verified, {state['body_bytes']} bytes", flush=True)
        state['status'] = 'complete' if len(state['complete']) == 128 else 'partial'
    except Exception as error:
        state.update(status='stopped', error=type(error).__name__)
    finally:
        state.pop('current', None)
        checkpoint()
        print(json.dumps(dict(status=state['status'], complete=len(state['complete']),
                              failed=len(state['failed']), body_bytes=state['body_bytes'])), flush=True)


if __name__ == '__main__':
    main()
