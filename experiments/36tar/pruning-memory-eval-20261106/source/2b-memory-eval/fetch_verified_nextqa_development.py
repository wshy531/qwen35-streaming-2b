"""Fetch only the 12 preplanned development assets; dual consistency checks.

Original-author ZIP directory supplies size/CRC32. Public redistribution supplies
pinned LFS SHA256. Neither a mirror filename nor a size match alone is accepted.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zlib
import httpx


def main():
    root=Path(__file__).resolve().parents[2]
    evidence=root/'runs/development_audit_20261017_02'
    plan=json.loads((evidence/'nextqa_mirror_preflight.json').read_text(encoding='utf-8'))
    if plan['status']!='metadata_matches_only_no_media_downloaded' or plan['sources']!=12:
        raise ValueError('Expected preflight plan required')
    if plan['total_bytes']>64*1024*1024 or plan['repository']!='jinyoungkim/NExT-GQA':
        raise ValueError('Unexpected source/budget')
    output=(root/'datasets/2b-memory-eval/nextgqa-dev12-hfverified').resolve()
    if not output.is_relative_to((root/'datasets/2b-memory-eval').resolve()):
        raise ValueError('Invalid output root')
    output.mkdir(exist_ok=False)
    files=[];received=0;start=time.monotonic();source=None
    try:
        with httpx.Client(timeout=httpx.Timeout(20,connect=10),follow_redirects=True,
                          headers={'Accept-Encoding':'identity'}) as client:
            for item in plan['files']:
                source=item['source_path']
                rel=PurePosixPath(source+'.mp4')
                if rel.is_absolute() or '..' in rel.parts or ':' in str(rel):
                    raise ValueError('Unsafe source path')
                target=output.joinpath(*rel.parts)
                if not target.resolve().is_relative_to(output):
                    raise ValueError('Unsafe media destination')
                target.parent.mkdir(parents=True,exist_ok=True)
                partial=target.with_suffix('.mp4.part')
                url=f'https://huggingface.co/datasets/{plan["repository"]}/resolve/{plan["revision"]}/{item["mirror_path"]}'
                digest=hashlib.sha256();crc=0;size=0
                with client.stream('GET',url) as response:
                    if response.status_code!=200:
                        raise ValueError('Unexpected media response status')
                    with partial.open('xb') as f:
                        for chunk in response.iter_bytes(65536):
                            received+=len(chunk);size+=len(chunk)
                            if received>64*1024*1024 or size>item['bytes'] or time.monotonic()-start>300:
                                raise TimeoutError('Development acquisition budget exceeded')
                            digest.update(chunk);crc=zlib.crc32(chunk,crc);f.write(chunk)
                if size!=item['bytes'] or digest.hexdigest()!=item['mirror_lfs_sha256'] or \
                   f'{crc&0xffffffff:08x}'!=item['official_zip_crc32'].lower():
                    raise ValueError('Mirror SHA256 or original ZIP size/CRC32 mismatch')
                partial.rename(target)
                files.append(dict(source_path=source,path=rel.as_posix(),local_path=str(target),
                    bytes=size,sha256=digest.hexdigest(),crc32=f'{crc&0xffffffff:08x}',
                    pinned_mirror_sha256_match=True,official_zip_size_crc_match=True,
                    question_ids=item['query_ids']))
                (output/'download_manifest.json').write_text(json.dumps(dict(
                    status='verified_against_pinned_mirror_and_official_zip_directory',
                    source_status=plan['source_status'],revision=plan['revision'],files=files),indent=2)+'\n',encoding='utf-8')
                print(json.dumps(dict(source=source,bytes=size,verified=True)),flush=True)
    except Exception as error:
        (output/'download_error.json').write_text(json.dumps(dict(
            error_type=type(error).__name__,source_path=source,completed_files=len(files),
            bytes_received=received,preserve_partials=True),indent=2)+'\n',encoding='utf-8')
        raise RuntimeError('Verified development acquisition stopped; see sanitized receipt') from None
    print(json.dumps(dict(files=len(files),bytes=received,elapsed_s=time.monotonic()-start)))


if __name__=='__main__':
    main()
