"""Package only verified new candidate media, and small acquisition evidence."""
import io
import json
import tarfile
from pathlib import Path
from acquire_media import verify
from install_bundle import sha

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/delayed_training_candidates_20261104_01'


def main():
    plan = json.loads((PREP/'acquisition_plan.json').read_text())
    receipt = json.loads((PREP/'media_acquisition.json').read_text())
    assert receipt['status']=='complete' and len(receipt['files'])==150
    assert receipt['plan_sha256']==sha(PREP/'acquisition_plan.json')
    assert json.loads((PREP/'media_audit.summary.json').read_text())['status']=='inventory_complete'
    payload = {}
    for row in plan['files']:
        local = ROOT/'download-cache/delayed-training-nextqa-v1'/row['mirror_path']
        assert verify(local, row)
        payload['datasets/delayed-training-nextqa-v1/'+row['mirror_path']] = local
    for name in ('acquire_media.py','audit_candidates.py','install_bundle.py','package_media.py','prepare_review.py'):
        rel = 'experiments/2b-delayed-readiness-v1/'+name
        payload[rel] = ROOT/rel
    for name in ('acquisition_plan.json','media_acquisition.json','media_download_started.json',
                 'media_audit.jsonl','media_audit.summary.json'):
        rel = 'runs/delayed_training_candidates_20261104_01/'+name
        payload[rel] = ROOT/rel
    manifest = {name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in payload.items()}
    archive = ROOT/'download-cache/delayed-training-nextqa-v1-transfer.tar'
    with tarfile.open(archive,'x:') as tar:
        data = json.dumps(manifest,indent=2).encode()
        member = tarfile.TarInfo('bundle_manifest.json'); member.size=len(data)
        tar.addfile(member,io.BytesIO(data))
        for name,path in payload.items():
            tar.add(path,arcname=name,recursive=False)
    with (PREP/'transfer_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
                       entries=manifest),f,indent=2)
    print(json.dumps(dict(files=len(manifest),bytes=archive.stat().st_size,sha256=sha(archive))))


if __name__=='__main__':
    main()
