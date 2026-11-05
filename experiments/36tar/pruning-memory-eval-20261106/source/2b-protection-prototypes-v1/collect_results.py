"""Collect only finalized small result files from this project's approved runs."""
import argparse,subprocess,shlex,json,hashlib,re,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[2]
REMOTE='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def collect(name):
    assert re.fullmatch(r'(isolated-recent|protect-stream|protect-channels)-[0-9]+',name)
    code='''import json,hashlib,sys
from pathlib import Path
r=Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar');p=r/'runs'/sys.argv[1]
assert p.resolve().is_relative_to((r/'runs').resolve()) and not p.is_symlink()
assert (p/'completed.json').is_file() or (p/'failed.json').is_file(),'run not finalized'
files=[]
for x in sorted(p.iterdir()):
 if x.name not in ('answers.jsonl','completed.json','failed.json','numerical.jsonl'):continue
 assert x.is_file() and not x.is_symlink() and x.stat().st_size<=96*2**20
 with x.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 files.append(dict(name=x.name,bytes=x.stat().st_size,sha256=h))
assert sum(x['bytes'] for x in files)<=96*2**20
print(json.dumps(dict(path=str(p),files=files)))
'''
    z=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ChaoSuan',REMOTE+'/envs/isolated-recent-v1/bin/python -c '+shlex.quote(code)+' '+shlex.quote(name)],text=True,capture_output=True)
    if z.returncode:raise RuntimeError(z.stderr)
    manifest=json.loads(z.stdout);dest=R/'runs'/name;dest.mkdir(exist_ok=True)
    for item in manifest['files']:
        target=dest/item['name']
        if not target.exists():subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',f"ChaoSuan:{manifest['path']}/{item['name']}",str(target)],check=True)
        assert target.stat().st_size==item['bytes'] and digest(target)==item['sha256'],item['name']
    manifest.update(verified=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (dest/'download_verified.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(dict(run=name,files=len(manifest['files']),bytes=sum(x['bytes'] for x in manifest['files']),verified=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');collect(p.parse_args().name)
