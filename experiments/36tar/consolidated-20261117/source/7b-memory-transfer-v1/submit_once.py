"""Explicit user GPU-window authorization; verify assets and submit once."""
import hashlib,json,os,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
MODEL=ROOT/'models/Qwen2.5-VL-7B-Instruct'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*2**20):h.update(b)
    return h.hexdigest()
def main():
    assert os.name=='posix'
    authorization=json.loads((PREP/'GPU_authorization.json').read_text())
    assert authorization['approved'] and time.time()<authorization['submit_before_unix']
    lock=json.loads((PREP/'execution_lock.json').read_text())
    for rel,digest in lock.items():assert sha(ROOT/rel)==digest,rel
    import transformers
    assert transformers.__version__=='5.2.0'
    installed=Path(transformers.__file__).parent
    wheel=PREP/'official_package/transformers-5.2.0-py3-none-any.whl'
    assert sha(wheel)=='9ecaf243dc45bee11a7d93f8caf03746accc0cb069181bbf4ad8566c53e854b4'
    with zipfile.ZipFile(wheel) as z:
        for name in ['modeling_qwen2_5_vl.py','processing_qwen2_5_vl.py','configuration_qwen2_5_vl.py']:
            rel='models/qwen2_5_vl/'+name
            assert (installed/rel).read_bytes()==z.read('transformers/'+rel)
    assert json.loads((PREP/'asset_completed.json').read_text())['files']==15
    manifest=json.loads((PREP/'asset_manifest.json').read_text())
    receipts={r['name']:r for r in json.loads((PREP/'asset_receipts.json').read_text())}
    verified=[]
    for item in manifest['files']:
        p=MODEL/item['name'];assert p.is_file() and p.stat().st_size==item['size']
        digest=sha(p);assert digest==receipts[item['name']]['sha256']
        if item['lfs_sha256']:assert digest==item['lfs_sha256']
        verified.append(dict(name=item['name'],size=p.stat().st_size,sha256=digest))
    assert not list(MODEL.glob('*.incoming-*'))
    script=ROOT/'experiments/7b-memory-transfer-v1/run_profile.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    assert time.time()<authorization['submit_before_unix'],'authorization window ended before submit'
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    cmd=['sbatch','--parsable','--no-requeue','--output=runs/7b-memory-slurm-%j.log','--error=runs/7b-memory-slurm-%j.log',str(script)]
    with (PREP/'submission_claim.json').open('x') as f:json.dump(dict(time=time.time(),command=cmd,lock_sha256=sha(PREP/'execution_lock.json')),f,indent=2)
    (PREP/'asset_pre_submit_verified.json').write_text(json.dumps(verified,indent=2)+'\n')
    result=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
    receipt=dict(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip(),automatic_retry=False)
    (PREP/'submission_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    assert result.returncode==0 and result.stdout.strip().split(';')[0].isdigit()
if __name__=='__main__':main()
