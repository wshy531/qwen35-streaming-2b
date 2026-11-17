import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_content_residual_prepare_20261113_01'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
lock=json.loads((PREP/'lock.json').read_text())
for f,h in lock.items():assert sha(ROOT/f)==h,f
plan=json.loads((PREP/'plan.json').read_text())
assert sha(ROOT/plan['media_path'])==plan['media_sha256']
print(json.dumps(dict(execution_files=len(lock),media=1,plan=plan['purpose'])))
