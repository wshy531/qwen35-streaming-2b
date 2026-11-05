"""Install public manifests and small review evidence in the authorized project."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from package_dataset import DEST, ROOT, read, save

REMOTE = '/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
OUT = ROOT / 'runs/dataset_delivery_20261105_01'


def ssh(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             'ChaoSuan', 'python -'], input=code, text=True, encoding='utf-8',
                            capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'cluster_sync_receipt.json').exists()
    assert not (OUT / 'metadata_and_review.tar.gz').exists()
    files = list(DEST.glob('*.json')) + list(DEST.glob('*.md'))
    files = [p for p in files if 'LOCAL_ONLY' not in p.name or p.name == 'bindings_LOCAL_ONLY.json']
    files += list((ROOT / 'experiments/eval-consolidation-20261105').glob('*.py'))
    review = ROOT / 'runs/diagnosis_expansion_review_20261105_01'
    files += [p for p in review.rglob('*') if p.is_file() and p.suffix in ('.json', '.jpg')
              and 'LOCAL_ONLY' not in p.name]
    files += [ROOT / 'runs/diagnosis_expansion_download_20261105_01/download.json']
    assert all(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink() for p in files)
    manifest = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert len(manifest) == len(files)
    assert not any('answer_keys' in name or 'notes_LOCAL_ONLY' in name for name in manifest)
    old = read(ROOT / 'runs/eval_consolidation_20261105_03/unified_package_cluster_sync.json')['files']
    implementations = read(DEST / 'execution_contract_template.json')['implementations']
    code = f'''from pathlib import Path
import hashlib,json
r=Path({REMOTE!r})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,value in {dict(old, **implementations)!r}.items():
 p=r/name
 assert p.is_file() and not p.is_symlink() and sha(p)==value, name
for name,value in {manifest!r}.items():
 p=r/name
 assert p.resolve().is_relative_to(r.resolve()) and not p.is_symlink(),name
 if p.exists() and name not in {list(old)!r}:assert sha(p)==value,name
d=r/'runs/dataset_delivery_20261105_01'
assert not d.exists()
d.mkdir()
print(json.dumps(dict(preflight=True)))
'''
    assert ssh(code)['preflight']
    archive = OUT / 'metadata_and_review.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(files):
            tar.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
    assert archive.stat().st_size < 64 * 2**20
    remote_archive = REMOTE + '/runs/dataset_delivery_20261105_01/' + archive.name
    subprocess.run(['scp', '-o', 'BatchMode=yes', str(archive), 'ChaoSuan:' + remote_archive],
                   check=True, timeout=180)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    code = f'''from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,os,time
r=Path({REMOTE!r});archive=Path({remote_archive!r});expected={manifest!r}
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert len(members)==len(expected) and {{m.name for m in members}}==set(expected)
 for m in members:
  p=r/m.name
  assert m.isfile() and '..' not in PurePosixPath(m.name).parts
  assert p.resolve().is_relative_to(r.resolve()) and not p.is_symlink()
  data=tar.extractfile(m).read()
  assert hashlib.sha256(data).hexdigest()==expected[m.name]
  p.parent.mkdir(parents=True,exist_ok=True)
  temporary=p.with_name(p.name+'.installing')
  assert not temporary.exists()
  with temporary.open('xb') as f:f.write(data)
  temporary.replace(p)
for name,value in expected.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==value,name
registry=json.loads((r/'datasets/streaming-evaluation-v1/registry.json').read_text())
bindings=json.loads((r/'datasets/streaming-evaluation-v1/bindings_LOCAL_ONLY.json').read_text())
media=set()
for entry in registry['suites'].values():
 for q in json.loads((r/'datasets/streaming-evaluation-v1'/entry['path']).read_text()):
  p=(Path(bindings['legacy_D18_readonly_root'])/(q['source_id']+'.mp4') if q.get('media_binding')=='legacy_D18_readonly_root' else r/q['media_path'])
  assert p.is_file(),str(p)
  media.add(str(p))
inventory=[]
for folder in sorted((r/'datasets').iterdir()):
 if not folder.is_dir() or folder.is_symlink():continue
 size=count=videos=0
 for parent,dirs,names in os.walk(folder,followlinks=False):
  dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]
  for name in names:
   p=Path(parent)/name
   if p.is_symlink():continue
   size+=p.stat().st_size;count+=1
   videos+=p.suffix.lower() in ('.mp4','.webm','.avi','.mov','.mkv')
 inventory.append(dict(directory=folder.name,bytes=size,files=count,media_files=videos))
result=dict(verified_files=len(expected),media_bindings=len(media),labels_uploaded=False,
 timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),dataset_inventory=inventory,
 datasets_total_bytes=sum(x['bytes'] for x in inventory),GPU=0)
(r/'runs/dataset_delivery_20261105_01/verification.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''
    result = ssh(code)
    save(OUT / 'cluster_sync_receipt.json', dict(remote_root=REMOTE, files=manifest,
         archive_sha256=digest, archive_bytes=archive.stat().st_size, remote_checks=result,
         labels_uploaded=False))
    print(dict(files=len(files), media_bindings=result['media_bindings'],
               archive_MiB=round(archive.stat().st_size/2**20,2),
               datasets_GiB=round(result['datasets_total_bytes']/2**30,3)))


if __name__ == '__main__':
    main()
