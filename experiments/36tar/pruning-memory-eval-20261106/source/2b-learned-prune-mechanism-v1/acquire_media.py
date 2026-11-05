"""One approved bounded acquisition. No implicit download or automatic retry."""
import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path, PurePosixPath
import httpx

REVISION = '5d1c262802efe7e5331629ca01e67916867590a5'
REPOSITORY = 'jinyoungkim/NExT-GQA'
BYTE_LIMIT = 400 * 1024**2


def verify(path, row):
    h = hashlib.sha256(); crc = 0; size = 0
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024**2), b''):
            size += len(data); h.update(data); crc = zlib.crc32(data, crc)
    return size == row['bytes'] and h.hexdigest() == row['mirror_lfs_sha256'] and f'{crc:08x}' == row['official_zip_crc32'].lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--download-approved', action='store_true')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    assert plan['repository'] == REPOSITORY and plan['revision'] == REVISION
    assert len(plan['files']) == 240 and sum(r['bytes'] for r in plan['files']) == plan['total_bytes'] == 359326839
    root = args.project_root.resolve()
    allowed = [Path('E:/VIT Model').resolve(), Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar').resolve()]
    assert root in allowed, 'project root not authorized'
    media_root = root / ('download-cache/learned-prune-nextqa-v1' if root == allowed[0] else 'datasets/learned-prune-nextqa-v1')
    targets = []
    for row in plan['files']:
        rel = PurePosixPath(row['mirror_path'])
        assert not rel.is_absolute() and '..' not in rel.parts and ':' not in str(rel)
        target = media_root / Path(*rel.parts)
        assert target.resolve().is_relative_to(media_root.resolve())
        targets.append(target)
    assert len(set(targets)) == 240
    print(json.dumps(dict(files=240, bytes=plan['total_bytes'], download=args.download_approved)), flush=True)
    if not args.download_approved:
        return
    out = root / 'runs/learned_prune_data_prepare_20261101_01'
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / 'media_acquisition.json'
    assert not receipt.exists(), 'existing attempt must be reviewed; no automatic retry'
    started = time.monotonic(); transferred = 0; records = []
    status = 'failed'; error = None
    try:
        with httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=True, headers={'Accept-Encoding': 'identity'}) as client:
            for row, target in zip(plan['files'], targets):
                if target.exists():
                    assert verify(target, row), 'existing media mismatch; no overwrite'
                    records.append(dict(video_id=row['video_id'], status='existing_verified'))
                    continue
                partial = target.with_suffix('.mp4.part')
                assert not partial.exists(), 'partial from earlier attempt; stop for review'
                target.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{row["mirror_path"]}'
                size = 0
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    assert response.status_code == 200
                    with partial.open('xb') as f:
                        for data in response.iter_bytes(65536):
                            transferred += len(data); size += len(data)
                            if transferred > BYTE_LIMIT or size > row['bytes'] or time.monotonic() - started > 45 * 60:
                                raise RuntimeError('acquisition budget exceeded')
                            f.write(data)
                assert verify(partial, row), 'SHA/size/official CRC mismatch'
                partial.rename(target)
                records.append(dict(video_id=row['video_id'], status='download_verified'))
                print(json.dumps(dict(complete=len(records), transferred_bytes=transferred)), flush=True)
        status = 'complete'
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        receipt.write_text(json.dumps(dict(status=status, error=error, files=records,
                                          transferred_bytes=transferred, seconds=time.monotonic()-started,
                                          plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest()), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
