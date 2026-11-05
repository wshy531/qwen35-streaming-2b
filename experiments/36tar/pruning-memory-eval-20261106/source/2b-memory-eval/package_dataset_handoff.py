"""Small code/docs/evidence handoff: excludes media, QA data, weights and envs."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    sources=[root/name for name in ('AGENTS.md','实验总纲.md','架构设计.md')]
    package=root/'experiments/2b-memory-eval'
    sources.extend(p for p in package.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md'))
    sources.extend(run/name for name in ['数据包说明与阶段结果.md','fixed_policy_actual_coverage.json',
        'source_overlap_screen.json','final_media_integrity.json','actual_memory_query_blueprints.json',
        'review_witness_input_binding.json',
        'cpu_delivery_checks.json',
        'overlap_engineering_correction.json','content_preflight_source_binding_correction.json'])
    sources.extend(run/'content_release_v1'/name for name in ['summary.json','media_inventory.json',
        'source_registry.json','content_hashes.json'])
    sources.extend(run/'handoff'/name for name in ['development_admission_remaining.json','review_worklist_receipt.json'])
    sources.append(run/'draft_input_export/receipt.json')
    # Coverage contains no full questions/options/answers, only IDs and spans.
    # Answer-bearing overlays/reviewer worklists are deliberately not archived.
    if len(set(sources))!=len(sources):raise ValueError('duplicate archive sources')
    now=datetime.now(timezone.utc);dest=root/'handoff';dest.mkdir(exist_ok=True)
    archive=dest/('2b-dataset-prep-'+now.strftime('%Y%m%dT%H%M%SZ')+'.zip')
    manifest=[]
    for p in sources:
        if not p.is_file() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('unexpected source')
        if p.stat().st_size>8*1024*1024:raise ValueError('small evidence budget exceeded')
        manifest.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,item in zip(sources,manifest):z.write(p,item['path'])
        z.writestr('FILE_MANIFEST.json',json.dumps(dict(captured_utc=now.isoformat(),files=manifest,
            excludes=['weights','video_or_RGB_media','raw_QA_or_answer_overlays','environments','keys','git_metadata'],
            main_progress_record='Original E:/VIT Model/实验总纲.md; bundled copy is timestamped evidence only.'),
            ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        for item in manifest:
            raw=z.read(item['path'])
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('archive member verification failed')
    receipt=dict(status='small_handoff_archive_member_hashes_verified',archive=str(archive),
        archive_bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_files=len(manifest),uncompressed_source_bytes=sum(x['bytes'] for x in manifest),
        media_or_raw_QA_included=False,captured_utc=now.isoformat())
    (run/'handoff_archive_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
