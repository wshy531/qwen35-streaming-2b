"""Extract only regular allowlisted new files under the verified project."""
from pathlib import Path, PurePosixPath
import hashlib,json,tarfile
ROOT=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar').resolve()
ALLOWED=Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert ROOT.is_relative_to(ALLOWED) and ROOT != ALLOWED
OUT=ROOT/'runs/2b_matched_coverage_prepare_20261023_01'
manifest=json.loads((OUT/'transfer_manifest.json').read_text())
with tarfile.open(OUT/'transfer.tar') as tar:
    members=tar.getmembers()
    assert len(members)==len(manifest) and {m.name for m in members}==set(manifest)
    entries=[]
    for m in members:
        p=PurePosixPath(m.name)
        assert not p.is_absolute() and '..' not in p.parts and m.isfile()
        assert m.name.startswith(('experiments/2b-matched-coverage-v1/','runs/2b_matched_coverage_prepare_20261023_01/inputs/')) or m.name.endswith('/LOCAL_CPU_preflight.json')
        target=(ROOT/m.name).resolve()
        assert target.is_relative_to(ROOT)
        data=tar.extractfile(m).read()
        assert hashlib.sha256(data).hexdigest()==manifest[m.name]
        if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==manifest[m.name],m.name
        entries.append((target,data))
    for target,data in entries:
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
lock=json.loads((OUT/'inputs/execution_lock.json').read_text())
for name,digest in lock['runtime_sha256'].items():
    p=(ROOT/name).resolve()
    assert p.is_relative_to(ROOT) and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(status='deployed_and_runtime_SHA_verified',transfer_files=len(manifest),runtime_files=len(lock['runtime_sha256']))))
