"""Acquire the 48 fixed non-development GQA assets, without changing splits.

Pinned public redistribution SHA256 plus original author ZIP size/CRC32 checks.
Does not visit the exhausted Drive branch, run models, or inspect answer quality.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import time
import zlib

import httpx

REPOSITORY='jinyoungkim/NExT-GQA'
REVISION='5d1c262802efe7e5331629ca01e67916867590a5'
MAX_BYTES=256*1024*1024


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--download-approved',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    sources=list(dict.fromkeys(r['source_path'] for r in rows if r['dataset']=='nextgqa' and r['split']!='development'))
    index_path=root/'runs/nextqa_zip_preflight_20261016_01/directory/archive_index.json'
    entries={v['path']:v for v in json.loads(index_path.read_text(encoding='utf-8'))['entries']}
    if not args.plan.exists():
        paths=['NExTVideo/'+s+'.mp4' for s in sources]
        with httpx.Client(timeout=30,follow_redirects=True) as client:
            response=client.post(f'https://huggingface.co/api/datasets/{REPOSITORY}/paths-info/{REVISION}',json={'paths':paths,'expand':False})
            response.raise_for_status()
            if len(response.content)>2*1024*1024:
                raise ValueError('metadata byte budget')
            mirror={v['path']:v for v in response.json()}
        files=[]
        for source,path in zip(sources,paths):
            a,b=entries[path],mirror[path]
            if a['bytes']!=b['size'] or not b.get('lfs',{}).get('oid'):
                raise ValueError('original ZIP / pinned mirror size or digest missing')
            qs=[r for r in rows if r['dataset']=='nextgqa' and r['source_path']==source]
            files.append(dict(source_path=source,mirror_path=path,bytes=a['bytes'],
                mirror_lfs_sha256=b['lfs']['oid'],official_zip_crc32=a['crc32'],
                split=qs[0]['split'],question_ids=[q['id'] for q in qs]))
        plan=dict(status='planned_not_downloaded',repository=REPOSITORY,revision=REVISION,
            source_status='third_party_redistribution_not_original_author_host',
            original_zip_directory_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            files=files,total_bytes=sum(v['bytes'] for v in files),
            official_crc32_is_consistency_not_cryptographic_authentication=True)
        args.plan.parent.mkdir(parents=True,exist_ok=True)
        args.plan.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['revision']!=REVISION or plan['repository']!=REPOSITORY or len(plan['files'])!=48 or plan['total_bytes']>MAX_BYTES or [v['source_path'] for v in plan['files']]!=sources:
        raise ValueError('fixed acquisition source / count / byte budget mismatch')
    print(json.dumps(dict(files=48,bytes=plan['total_bytes'],download=args.download_approved)),flush=True)
    if not args.download_approved:
        return
    output=args.out.resolve()
    if not output.is_relative_to((root/'datasets/2b-memory-eval').resolve()):
        raise ValueError('destination outside project dataset root')
    output.mkdir(parents=True,exist_ok=False)
    done=[];received=0;start=time.monotonic();partial=None;source=None
    try:
        with httpx.Client(timeout=httpx.Timeout(30,connect=10),follow_redirects=True,
                          headers={'Accept-Encoding':'identity'}) as client:
            for entry in plan['files']:
                source=entry['source_path'];rel=PurePosixPath(source+'.mp4')
                if rel.is_absolute() or '..' in rel.parts or ':' in str(rel):
                    raise ValueError('unsafe relative path')
                target=output.joinpath(*rel.parts)
                if not target.resolve().is_relative_to(output):raise ValueError('escaped output')
                target.parent.mkdir(parents=True,exist_ok=True)
                partial=target.with_suffix('.mp4.part');h=hashlib.sha256();crc=0;size=0
                url=f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{entry["mirror_path"]}'
                with client.stream('GET',url) as response:
                    if response.status_code!=200:raise ValueError('unexpected media status')
                    with partial.open('xb') as handle:
                        for chunk in response.iter_bytes(65536):
                            received+=len(chunk);size+=len(chunk)
                            if received>MAX_BYTES or size>entry['bytes'] or time.monotonic()-start>1500:
                                raise TimeoutError('acquisition budget reached')
                            h.update(chunk);crc=zlib.crc32(chunk,crc);handle.write(chunk)
                if size!=entry['bytes'] or h.hexdigest()!=entry['mirror_lfs_sha256'] or f'{crc&0xffffffff:08x}'!=entry['official_zip_crc32'].lower():
                    raise ValueError('SHA256 / original ZIP size CRC32 mismatch')
                partial.rename(target);partial=None
                done.append(dict(source_path=source,local_path=str(target),bytes=size,
                    sha256=h.hexdigest(),crc32=f'{crc&0xffffffff:08x}',split=entry['split'],
                    question_ids=entry['question_ids'],mirror_sha_and_author_zip_consistency_verified=True))
                (output/'download_manifest.json').write_text(json.dumps(dict(
                    status='downloaded_consistency_verified_not_visually_reviewed',repository=REPOSITORY,
                    revision=REVISION,source_status=plan['source_status'],files=done),indent=2)+'\n',encoding='utf-8')
                print(json.dumps(dict(source=source,completed=len(done),bytes=size,verified=True)),flush=True)
    except Exception as error:
        (output/'download_error.json').write_text(json.dumps(dict(
            status='stopped_no_automatic_retry',error_type=type(error).__name__,source_path=source,
            bytes_received=received,completed=len(done),partial_file=str(partial) if partial else None),indent=2)+'\n',encoding='utf-8')
        raise RuntimeError('bounded acquisition stopped; see sanitized receipt') from None
    print(json.dumps(dict(completed=len(done),bytes=received,elapsed_s=time.monotonic()-start)))


if __name__=='__main__':main()
