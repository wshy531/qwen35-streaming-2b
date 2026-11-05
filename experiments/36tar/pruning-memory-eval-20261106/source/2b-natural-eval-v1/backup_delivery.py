"""Small text-only handoff; exclude media, model assets, environments and keys."""
import base64,hashlib,json,re,zipfile
from pathlib import Path
from datetime import datetime,timezone
from cluster_io import command,LOCAL,HERE
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    roots=[HERE,LOCAL,ROOT/'runs/2b_early9_diagnostic_20261020_01',ROOT/'runs/2b_temporal_route_20261020_01',ROOT/'runs/early25_full_prefix_review_20261020_01']
    selected=[ROOT/n for n in ['实验总纲.md','架构设计.md','2B自然视频与长短期记忆首轮结果.md','早期25题全前缀审核结果.md']]
    allowed={'.py','.sh','.md','.json','.jsonl','.csv','.txt','.log'}
    for folder in roots:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts and p.name not in ['backup_receipt.json','backup_upload_receipt.json']:
                if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('unsafe backup member')
                selected.append(p)
    selected=sorted(set(selected));total=sum(p.stat().st_size for p in selected)
    if total>64*2**20 or any(p.stat().st_size>16*2**20 for p in selected):raise ValueError('bounded text backup exceeded')
    entries=[]
    for p in selected:
        value=p.read_bytes()
        if re.search(rb'(?m)^-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----',value):raise ValueError('credential-like material excluded')
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
    manifest=dict(created_UTC=datetime.now(timezone.utc).isoformat(),scope='project text/code/config/small results only; no weights/videos/images/env/keys',master_is_snapshot_before_backup_receipt=True,files=entries)
    folder=ROOT/'handoff';folder.mkdir(exist_ok=True);name='2b-natural-memory-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip';out=folder/name
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in selected:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        if set(z.namelist())!={e['path'] for e in entries}|{'MANIFEST.json'}:raise ValueError('ZIP member mismatch')
        for e in entries:
            if hashlib.sha256(z.read(e['path'])).hexdigest()!=e['sha256']:raise ValueError('backup byte identity')
    receipt=dict(local_path=str(out),zip_bytes=out.stat().st_size,zip_sha256=sha(out),files=len(entries),members=len(entries)+1,raw_text_bytes=total,CRC_passed=True,all_original_bytes_match=True)
    (LOCAL/'backup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
def upload():
    receipt=json.loads((LOCAL/'backup_receipt.json').read_text());p=Path(receipt['local_path'])
    if sha(p)!=receipt['zip_sha256'] or p.stat().st_size>16*2**20:raise ValueError('bounded backup identity')
    payload=base64.b64encode(p.read_bytes()).decode()
    code='name='+repr(p.name)+'\nexpected='+repr(receipt['zip_sha256'])+'\npayload='+repr(payload)+r'''
import pathlib,base64,hashlib,json
p=pathlib.Path.cwd();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
if not p.resolve().is_relative_to(allowed):raise ValueError('wrong project')
folder=p/'handoff';folder.mkdir(exist_ok=True);dest=folder/name
if not dest.resolve().is_relative_to(p.resolve()):raise ValueError('output escape')
value=base64.b64decode(payload)
if hashlib.sha256(value).hexdigest()!=expected:raise ValueError('backup hash')
with dest.open('xb') as f:f.write(value)
if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected:raise ValueError('remote backup integrity')
print(json.dumps(dict(remote_path=str(dest),bytes=len(value),sha256=expected,status='remote_backup_verified')))
'''
    s=command(code,timeout=120);(LOCAL/'backup_upload_receipt.json').write_text(s,encoding='utf-8');print(s)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['create','upload']);a=p.parse_args();(main if a.mode=='create' else upload)()
