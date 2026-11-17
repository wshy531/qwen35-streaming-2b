"""Explicit file list only. No scanning of shared directories or model loading."""
import os,sys,zipfile,subprocess
from common import *

def main():
    assert os.name=='posix';verify()
    roots={}
    for q in read(PREP/'evaluation.json'):roots[str(path(q))]=q['media_sha256']
    for g in read(PREP/'fit_sources.json'):roots[str(path(g))]=g['media_sha256']
    checked=[]
    for p,h in sorted(roots.items()):
        assert sha(p)==h,p
        checked.append(dict(path=p,sha256=h,bytes=Path(p).stat().st_size))
    import transformers
    assert transformers.__version__=='5.2.0'
    installed=Path(transformers.__file__).parent
    old=ROOT/'runs/7b_transfer_prepare_20261107_01'
    with zipfile.ZipFile(old/'official_package/transformers-5.2.0-py3-none-any.whl') as z:
        for name in ('modeling_qwen2_5_vl.py','processing_qwen2_5_vl.py','configuration_qwen2_5_vl.py'):
            rel='models/qwen2_5_vl/'+name
            assert (installed/rel).read_bytes()==z.read('transformers/'+rel)
    assets=[]
    for r in read(old/'asset_receipts.json'):
        p=ROOT/'models/Qwen2.5-VL-7B-Instruct'/r['name'];assert sha(p)==r['sha256'];assets.append(r['name'])
    for mode in ('prune','selector'):
        script=Path(__file__).parent/f'run_{mode}.sh'
        assert script.read_bytes().startswith(b'#!/bin/bash\n') and b'\r' not in script.read_bytes(), 'Slurm requires UNIX line endings'
        subprocess.run(['bash','-n',str(script)],check=True)
    save(PREP/'remote_verified.json',dict(passed=True,media=checked,model_assets=assets,official_source_exact=True,lock_sha256=sha(PREP/'lock.json')))
    print(json.dumps(dict(passed=True,media=len(checked),assets=len(assets))))
if __name__=='__main__':main()
