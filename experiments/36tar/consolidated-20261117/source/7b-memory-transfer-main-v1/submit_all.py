"""One authorized four-part chain; at most one A800, no automatic retry."""
import hashlib,json,os,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/7b_formal_execution_20261107_01'
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(8*2**20):h.update(block)
    return h.hexdigest()
def main():
    assert os.name=='posix'
    auth=read(RUN/'authorization.json')
    assert auth['approved'] and auth['parts']==4 and auth['minutes_per_part']==120 and auth['max_concurrent_gpu']==1
    for name in (RUN/'execution_lock.json',PREP/'execution_lock.json'):
        for rel,digest in read(name).items():assert sha(ROOT/rel)==digest,rel
    assert read(RUN/'CPU_cluster.json')['media_verified']==201
    import transformers
    assert transformers.__version__=='5.2.0'
    wheel=PREP/'official_package/transformers-5.2.0-py3-none-any.whl'
    assert sha(wheel)=='9ecaf243dc45bee11a7d93f8caf03746accc0cb069181bbf4ad8566c53e854b4'
    with zipfile.ZipFile(wheel) as z:
        for name in ('modeling_qwen2_5_vl.py','processing_qwen2_5_vl.py','configuration_qwen2_5_vl.py'):
            rel='models/qwen2_5_vl/'+name
            assert (Path(transformers.__file__).parent/rel).read_bytes()==z.read('transformers/'+rel)
    model=ROOT/'models/Qwen2.5-VL-7B-Instruct'
    verified=[]
    for r in read(PREP/'asset_receipts.json'):
        p=model/r['name'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['name']
        verified.append(r)
    assert len(verified)==15
    script=ROOT/'experiments/7b-memory-transfer-main-v1/run_batch.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    helptext=subprocess.run(['sbatch','--help'],check=True,text=True,capture_output=True).stdout
    assert '--dependency' in helptext
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    with (RUN/'submission_claim.json').open('x') as f:json.dump(dict(time=time.time(),parts=4,execution_lock_sha256=sha(RUN/'execution_lock.json')),f,indent=2)
    save(RUN/'assets_reverified.json',verified)
    receipts=[];previous=None
    for part in range(1,5):
        cmd=['sbatch','--parsable','--no-requeue','--output=runs/7b-transfer-main-%j.log','--error=runs/7b-transfer-main-%j.log']
        if previous is not None:cmd+=['--dependency=afterok:'+previous]
        cmd+=[str(script),str(part)]
        result=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
        r=dict(part=part,command=cmd,returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip(),automatic_retry=False)
        receipts.append(r);save(RUN/'submission_receipts.json',receipts);print(json.dumps(r),flush=True)
        assert result.returncode==0 and result.stdout.strip().split(';')[0].isdigit()
        previous=result.stdout.strip().split(';')[0]
if __name__=='__main__':main()
