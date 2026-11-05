"""One specifically approved auxiliary batch; no retries or GPU use."""
import datetime, hashlib, json, os, subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2]
P=R/'runs/evidence-scale-nextqa-analysis-20261102-01'
O=R/'runs/evidence-scale-nextqa-judge-20261102-01'
expected='00ed0495472887198ce90c01088d476a1a0c9a9c380a9be5d3ed3fe510357438'
assert not O.exists()
assert hashlib.sha256((P/'judge_execution_lock.json').read_bytes()).hexdigest()==expected
for name,digest in json.loads((P/'judge_execution_lock.json').read_text()).items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
auth=dict(approved=True,cap_cny=15,max_requests=163,lock_sha256=expected,
          approved_scope='User explicitly approved this 15 yuan auxiliary batch on 2026-11-02',recorded_utc=now)
with (P/'authorization.json').open('x',encoding='utf-8') as f:json.dump(auth,f,indent=2)
with (P/'launch_intent.json').open('x',encoding='utf-8') as f:json.dump(dict(time=now,output=str(O)),f)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
with (P/'judge_console.log').open('xb') as log:
    proc=subprocess.Popen([str(R/'envs/evidence-scale-v1/bin/python'),str(R/'experiments/2b-evidence-scale-v1/judge_batch.py'),
                           '--inputs',str(P),'--out',str(O)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,started_utc=now,output=str(O),cap_cny=15,max_requests=163,GPU=0,automatic_retry=False)
(P/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
