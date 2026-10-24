"""Small verified handoff: source, configuration and CPU evidence only."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    paths=[ROOT/'实验总纲.md',ROOT/'架构设计.md']
    paths += [p for p in (ROOT/'experiments/2b-ls-depth-v1').iterdir() if p.suffix in ('.py','.sh','.md')]
    paths += [p for p in (HERE/'inputs').iterdir() if p.suffix=='.json']
    paths += [HERE/name for name in ('CPU_preflight.json','cluster_CPU_execution.json','final_cluster_checks.json',
        'deploy_receipt.json','analysis_lock.json','scorer_contract.json','storage_accounting.json',
        'historical_references_LOCAL_ONLY.json','input_provenance.json','labels_LOCAL_ONLY.json')]
    paths += list(HERE.glob('*.py'))
    paths=list(dict.fromkeys(paths))
    for path in paths:
        assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
        assert path.suffix in ('.py','.sh','.md','.json') and 'authorization' not in path.name.lower()
        assert 'approval' not in path.name.lower() and 'submit_attempt' not in path.name.lower()
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
    total=sum(p.stat().st_size for p in paths);assert total<8*2**20
    target=ROOT/'handoff/2b-ls-depth-CPU-prepare-20261021.zip'
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in paths}
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:archive.write(path,arcname=path.relative_to(ROOT).as_posix())
        archive.writestr('PREPARE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,record in manifest.items():
            data=archive.read(name);assert len(data)==record['bytes'] and sha(data)==record['sha256']
    receipt=dict(files=len(paths),input_bytes=total,archive_bytes=target.stat().st_size,
        archive_sha256=sha(target.read_bytes()),member_SHA_and_CRC_passed=True,
        excludes_weights_media_environments_credentials_and_live_GPU_approvals=True)
    with (HERE/'handoff_receipt.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
    print(receipt)


if __name__=='__main__':main()
