"""Bounded metadata probe and separately authorized single-pass video fetch."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
import zlib

import httpx

from prepare import OUT, ROOT, read, save, sha

MEDIA = ROOT / 'datasets/2b-memory-eval/T-new40-20261020'
REPO = 'jinyoungkim/NExT-GQA'
REV = '5d1c262802efe7e5331629ca01e67916867590a5'
BYTE_CAP = 256 * 2**20


def frozen():
    receipt = read(OUT/'metadata_freeze.json')
    for path, digest in receipt['input_sha256'].items():
        assert sha(ROOT/path) == digest, path
    for name, digest in receipt['output_sha256'].items():
        assert sha(OUT/name) == digest, name
    proposal = read(OUT/'acquisition_proposal.json')
    assert proposal['repository'] == REPO and proposal['revision'] == REV
    assert len(proposal['sources']) == 40
    assert proposal['total_media_bytes'] == 243303701 < BYTE_CAP
    return proposal


def metadata():
    proposal = frozen()
    save(OUT/'metadata_probe_started.json', dict(
        source_sha256=sha(__file__), attempts=1, automatic_retries=0,
        media_bytes=0, metadata_byte_cap=2*2**20, wall_seconds=90))
    paths = [r['official_archive_path'] for r in proposal['sources']]
    started = time.monotonic()
    data = bytearray()
    try:
        with httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=True) as client:
            with client.stream('POST', f'https://huggingface.co/api/datasets/{REPO}/paths-info/{REV}',
                               json={'paths':paths, 'expand':False}) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(65536):
                    data.extend(chunk)
                    if len(data) > 2*2**20 or time.monotonic()-started > 90:
                        raise TimeoutError('metadata budget')
        rows = json.loads(data)
        mirror = {r['path']:r for r in rows}
        assert len(rows) == len(mirror) == 40 and set(mirror) == set(paths)
        files = []
        for row in proposal['sources']:
            m = mirror[row['official_archive_path']]
            digest = m['lfs']['oid']
            assert row['bytes'] == m['size']
            assert re.fullmatch('[0-9a-f]{64}', digest)
            files.append(dict(**row, mirror_sha256=digest))
        save(OUT/'acquisition_plan.json', dict(
            status='mirror_metadata_verified_NOT_media_acquired', files=files,
            repository=REPO, revision=REV, total_media_bytes=sum(r['bytes'] for r in files),
            source_sha256=sha(__file__), proposal_sha256=sha(OUT/'acquisition_proposal.json'),
            candidate_sha256=sha(OUT/'candidate_questions119.json'),
            media_byte_cap=BYTE_CAP, metadata_bytes=len(data), metadata_seconds=time.monotonic()-started,
            wall_seconds=2700, maximum_attempts=1, automatic_retries=0,
            GPU_jobs=0, cluster_operations=0, authorization_required=True))
    except Exception as error:
        save(OUT/'metadata_probe_error.json', dict(type=type(error).__name__,
            metadata_bytes=len(data), seconds=time.monotonic()-started,
            media_bytes=0, no_automatic_retry=True))
        raise RuntimeError('Bounded metadata probe stopped; sanitized receipt saved') from None
    print(json.dumps(dict(metadata_verified=40, bytes=sum(r['bytes'] for r in files),
                          metadata_bytes=len(data), media_downloaded=0)), flush=True)


def download():
    frozen()
    plan = read(OUT/'acquisition_plan.json')
    auth = read(OUT/'download_authorization.json')
    assert auth['approved'] is True and auth['acquisition_plan_sha256'] == sha(OUT/'acquisition_plan.json')
    assert auth['media_byte_cap'] == plan['media_byte_cap'] == BYTE_CAP
    assert auth['wall_seconds'] == 2700
    assert sha(__file__) == plan['source_sha256']
    assert plan['candidate_sha256'] == sha(OUT/'candidate_questions119.json')
    assert plan['repository'] == REPO and plan['revision'] == REV
    assert MEDIA.resolve().is_relative_to((ROOT/'datasets/2b-memory-eval').resolve())
    MEDIA.mkdir(exist_ok=False)
    save(OUT/'download_started.json', dict(source_sha256=sha(__file__),
        authorization_sha256=sha(OUT/'download_authorization.json'), maximum_attempts=1,
        no_automatic_retry=True, media_byte_cap=BYTE_CAP, wall_seconds=2700))
    total = 0
    done = []
    partial = None
    current_source = None
    started = time.monotonic()
    try:
        with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=True,
                          headers={'Accept-Encoding':'identity'}) as client:
            for row in plan['files']:
                current_source = row['source_path']
                relative = PurePosixPath(current_source + '.mp4')
                assert not relative.is_absolute() and '..' not in relative.parts and ':' not in str(relative)
                target = MEDIA.joinpath(*relative.parts)
                assert target.resolve().is_relative_to(MEDIA.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix('.mp4.part')
                size, crc = 0, 0
                digest = hashlib.sha256()
                url = f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{row["official_archive_path"]}'
                with client.stream('GET', url) as response:
                    assert response.status_code == 200, 'unexpected media HTTP status'
                    with partial.open('xb') as stream:
                        for chunk in response.iter_bytes(65536):
                            size += len(chunk)
                            total += len(chunk)
                            if size > row['bytes'] or total > BYTE_CAP or time.monotonic()-started > 2700:
                                raise TimeoutError('download budget')
                            digest.update(chunk)
                            crc = zlib.crc32(chunk, crc)
                            stream.write(chunk)
                actual_crc = f'{crc & 0xffffffff:08x}'
                assert size == row['bytes']
                assert digest.hexdigest() == row['mirror_sha256']
                assert actual_crc == row['official_zip_crc32'].lower()
                partial.rename(target)
                partial = None
                done.append(dict(source_path=current_source, source_id=row['source_id'],
                    source_group_id=row['source_group_id'], media_path=target.relative_to(ROOT).as_posix(),
                    bytes=size, sha256=digest.hexdigest(), crc32=actual_crc,
                    question_ids=row['question_ids'], media_content_reviewed=False))
                # Distinct checkpoint per asset; never overwrite evidence on failure.
                save(OUT/f'download_checkpoint_{len(done):02d}.json', dict(files=done, bytes_received=total))
                print(json.dumps(dict(completed=len(done), source=current_source, bytes=size)), flush=True)
    except Exception as error:
        cleaned = False
        if partial is not None and partial.exists():
            assert partial.resolve().is_relative_to(MEDIA.resolve()) and partial.suffix == '.part'
            partial.unlink()
            cleaned = True
        save(OUT/'download_error.json', dict(type=type(error).__name__, source=current_source,
            files_verified=len(done), bytes_received=total, partial_removed=cleaned,
            seconds=time.monotonic()-started, no_automatic_retry=True))
        raise RuntimeError('Single download attempt stopped; sanitized receipt saved') from None
    assert len(done) == 40 and total == plan['total_media_bytes']
    save(OUT/'download_manifest.json', dict(
        status='all40_integrity_verified_NOT_content_admitted', files=done,
        bytes_received=total, seconds=time.monotonic()-started,
        acquisition_plan_sha256=sha(OUT/'acquisition_plan.json'),
        candidate_sha256=plan['candidate_sha256'], GPU_jobs=0, cluster_operations=0,
        model_answers=0, content_reviews=0))
    print(json.dumps(dict(downloaded=40, bytes=total, seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    {'metadata':metadata, 'download':download}[sys.argv[1]]()
