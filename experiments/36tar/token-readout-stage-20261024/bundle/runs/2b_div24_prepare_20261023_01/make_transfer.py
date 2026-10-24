"""Small allowlisted transfer; no model/media/env or remote GPU submission."""
from pathlib import Path, PurePosixPath
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

files={p.relative_to(ROOT).as_posix():p for p in (ROOT/'experiments/2b-div24-v1').iterdir()
       if p.suffix in ('.py','.sh','.md')}
files.update({p.relative_to(ROOT).as_posix():p for p in (OUT/'inputs').iterdir() if p.is_file()})
files[(OUT.relative_to(ROOT)/'LOCAL_CPU_preflight.json').as_posix()]=OUT/'CPU_preflight.json'
manifest={name:sha(p) for name,p in files.items()}
(OUT/'transfer_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
with tarfile.open(OUT/'transfer.tar','w') as tar:
    for name,p in files.items():tar.add(p,arcname=name,recursive=False)
print(dict(files=len(files),bytes=(OUT/'transfer.tar').stat().st_size))
