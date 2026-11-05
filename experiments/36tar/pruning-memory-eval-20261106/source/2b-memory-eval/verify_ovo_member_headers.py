"""Read only the 32 pinned 512-byte TAR headers after bounded extraction."""
import json
from pathlib import Path
import tarfile

from index_ovo_tar import HeaderReader,relative_name


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    prior=root/'runs/data_acquisition_preflight_20261016_01'
    plan=json.loads((run/'ovo32_plan.json').read_text())
    parts=sorted((r for r in json.loads((prior/'ovo_file_tree.json').read_text()) if r['path'].startswith('src_videos.tar.part')),key=lambda r:r['path'])
    reader=HeaderReader(parts,plan['revision'],120,max_bytes=32768,max_requests=64,max_read=512)
    entries=[]
    try:
        for item in plan['videos']:
            header=tarfile.TarInfo.frombuf(reader.read(item['header_offset'],512),'utf-8','surrogateescape')
            if relative_name(header.name)!=item['relative_path'] or header.size!=item['bytes'] or not header.isfile():
                raise ValueError('actual TAR header differs from saved index')
            entries.append(dict(source_path=item['relative_path'],size_and_tar_checksum_valid=True))
    finally:reader.client.close()
    (run/'ovo32_actual_header_checks.json').write_text(json.dumps(dict(status='all_planned_member_headers_verified',
        entries=entries,bytes_received=reader.bytes,requests=reader.requests,
        full_archive_hash_recomputed=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(headers=len(entries),bytes_received=reader.bytes,requests=reader.requests)))


if __name__=='__main__':main()
