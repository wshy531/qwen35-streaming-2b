"""Bind verified identical question inputs and bounded local CPU evidence."""
from pathlib import Path
import json,ast
from common import ROOT,OUT,ARMS,read,save,sha
HERE=Path(__file__).parent;run=OUT/'inputs'
for name in ('groups.json','questions.json','snapshots34.json'):
    assert sha(run/name)==sha(ROOT/'runs/2b_temporal24_prepare_20261023_01/inputs'/name)
cpu=read(OUT/'CPU_preflight.json');tiny=read(OUT/'tiny_session_check.json')
assert cpu['status']==tiny['status']=='passed' and cpu['layouts']==2727 and len(tiny['QA'])==9
if 'protocol_sha256' not in cpu:
    save(OUT/'CPU_preflight_before_protocol_binding.json',cpu)
    cpu['protocol_sha256']=sha(run/'protocol.json')
    cpu['protocol_binding_note']='post-preflight bind after verifying all question/frame/snapshot files identical to tested inputs'
    (OUT/'CPU_preflight.json').write_text(json.dumps(cpu,indent=2)+'\n',encoding='utf-8')
lock=read(ROOT/'runs/2b_temporal24_prepare_20261023_01/inputs/execution_lock.json')['runtime_sha256']
for rel,digest in lock.items():
    path=ROOT/rel if not rel.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
    assert sha(path)==digest,rel
for p in HERE.iterdir():
    if p.suffix in ('.py','.sh'):
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        lock[p.relative_to(ROOT).as_posix()]=sha(p)
save(run/'execution_lock.json',dict(protocol_sha256=sha(run/'protocol.json'),
    input_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},runtime_sha256=lock,
    local_CPU_sha256=sha(OUT/'CPU_preflight.json'),local_tiny_sha256=sha(OUT/'tiny_session_check.json'),
    no_GPU_authorization=True,declared_GPU_proposal=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,
    host_memory_GiB=24,wall_seconds=3600,worker_seconds=3300,answers_max=2727,generated_tokens_max=21816,
    visual_encodings=3981,pretrained_loads=1,log_bytes_cap=96*2**20,no_auto_retry=True)))
old=ROOT/'runs/2b_ls_depth_prepare_20261021_01/results-217894/answers.jsonl'
references=[]
for line in old.read_text(encoding='utf-8').splitlines():
    r=json.loads(line)
    if r['arm'] in ('M0','M2','A06','LS'):
        references.append(dict(id=r['id'],arm=r['arm']+'_full48',generated_ids=r['generated_ids'],
            input_ids_sha256=r.get('input_ids_sha256'),node_digest=r.get('node_digest')))
assert len(references)==1212
save(OUT/'historical_references_LOCAL_ONLY.json',dict(records=references,source_SHA256=sha(old)))
save(OUT/'analysis_lock.json',dict(scorer_SHA256=sha(HERE/'analyze.py'),labels_SHA256=sha(OUT/'labels_LOCAL_ONLY.json'),
    historical_references_SHA256=sha(OUT/'historical_references_LOCAL_ONLY.json')))
print(dict(status='frozen',runtime_files=len(lock),execution_lock_SHA256=sha(run/'execution_lock.json'),GPU_authorized=False))
