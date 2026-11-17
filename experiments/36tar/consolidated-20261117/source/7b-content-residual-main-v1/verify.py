import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_content_residual_main_prepare_20261113_01'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def verify(media=False):
    lock=json.loads((PREP/'lock.json').read_text())
    for f,h in lock.items():assert sha(ROOT/f)==h,f
    rows=json.loads((PREP/'media.json').read_text()) if media else []
    for r in rows:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    print(json.dumps(dict(verified_files=len(lock),verified_media=len(rows))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--media',action='store_true');a=p.parse_args();verify(a.media)
