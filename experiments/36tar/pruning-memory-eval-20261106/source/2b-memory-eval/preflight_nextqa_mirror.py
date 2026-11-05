"""One metadata-only check of a public redistribution, against official ZIP directory.

Does not download any video. A matching size is NOT a matching video digest.
"""
import hashlib
import json
from pathlib import Path
import time
import httpx


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'runs/development_audit_20261017_02'
    rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    archive_path=root/'runs/nextqa_zip_preflight_20261016_01/directory/archive_index.json'
    archive=json.loads(archive_path.read_text(encoding='utf-8'))
    entries={r['path']:r for r in archive['entries']}
    sources=list(dict.fromkeys(r['source_path'] for r in rows
        if r['dataset']=='nextgqa' and r['split']=='development'))
    repo='jinyoungkim/NExT-GQA'
    paths=['NExTVideo/'+s+'.mp4' for s in sources]
    start=time.monotonic()
    def read_json(response):
        response.raise_for_status()
        if len(response.content)>2*1024*1024 or time.monotonic()-start>90:
            raise ValueError('metadata budget exceeded')
        return response.json()
    with httpx.Client(timeout=30,follow_redirects=True) as client:
        info=read_json(client.get('https://huggingface.co/api/datasets/'+repo))
        revision=info['sha']
        if info.get('gated'):
            raise ValueError('public ungated source required; no credentials requested')
        files=read_json(client.post(f'https://huggingface.co/api/datasets/{repo}/paths-info/{revision}',
                                    json={'paths':paths,'expand':False}))
    indexed={f['path']:f for f in files}
    planned=[]
    for source,path in zip(sources,paths):
        original=entries[path]
        mirror=indexed[path]
        if original['bytes'] != mirror['size'] or not mirror.get('lfs',{}).get('oid'):
            raise ValueError('mirror entry size or LFS identity unavailable')
        planned.append(dict(source_path=source,mirror_path=path,bytes=original['bytes'],
            official_zip_crc32=original['crc32'],mirror_lfs_sha256=mirror['lfs']['oid'],
            query_ids=[r['id'] for r in rows if r['dataset']=='nextgqa' and
                       r['split']=='development' and r['source_path']==source]))
    plan=dict(status='metadata_matches_only_no_media_downloaded',repository=repo,revision=revision,
        source_status='third_party_redistribution_not_original_author_host',
        original_archive_directory_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        files=planned,total_bytes=sum(r['bytes'] for r in planned),
        sources=len(planned),questions=sum(len(r['query_ids']) for r in planned),
        required_before_acceptance=['local_SHA256_matches_pinned_mirror_LFS',
            'local_size_and_CRC32_match_official_ZIP_member',
            'decode_duration_and_time_alignment_against_author_annotations',
            'visual_answer_and_source_review_before_dataset_freeze'],
        caveat='Official ZIP CRC32 is a consistency check, not cryptographic authentication. Do not call redistribution the official host or accept a filename/size match as sufficient.')
    (out/'nextqa_mirror_preflight.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in plan.items() if k!='files'},ensure_ascii=False))


if __name__=='__main__':
    main()
