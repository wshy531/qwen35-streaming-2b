"""Bind C2 protocol, code, primary scorer and pre-existing odd24 references."""
import ast,json,re
from pathlib import Path
from common import ROOT,OUT,read,save,sha,ARMS
HERE=Path(__file__).parent
cpu=read(OUT/'CPU_preflight.json');run=OUT/'inputs'
assert cpu['status']=='passed' and cpu['protocol_sha256']==sha(run/'protocol.json')
old=ROOT/'runs/2b_div24_prepare_20261023_01'
lock=read(old/'inputs/execution_lock.json')['runtime_sha256']
for rel,digest in lock.items():
    p=ROOT/rel if not rel.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
    assert sha(p)==digest,rel
for p in HERE.iterdir():
    if p.suffix in ('.py','.sh'):
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        lock[p.relative_to(ROOT).as_posix()]=sha(p)
save(run/'execution_lock.json',dict(protocol_sha256=sha(run/'protocol.json'),
    input_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},runtime_sha256=lock,
    local_CPU_sha256=sha(OUT/'CPU_preflight.json'),no_GPU_authorization=True,
    declared_GPU_proposal=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,
        host_memory_GiB=24,wall_seconds=3600,worker_seconds=3300,pretrained_loads=1,
        answers_max=909,generated_tokens_max=7272,visual_encodings=3981,
        log_bytes_cap=64*2**20,no_training=True,no_download=True,no_auto_retry=True)))
source=old/'results-218021/answers.jsonl'
references=[dict(id=r['id'],arm='odd24',generated_ids=r['generated_ids'])
    for r in (json.loads(line) for line in source.read_text().splitlines()) if r['arm']=='odd24']
assert len(references)==303
save(OUT/'historical_references_LOCAL_ONLY.json',dict(records=references,source_SHA256=sha(source)))
proxies=[]
for q in read(run/'questions.json'):
    text=q['question'].lower()
    proxies.append(dict(id=q['id'],
        detail_question_words=bool(re.search(r'\b(colou?r|wear|shirt|hat|holding|hand|left|right|where)\b',text)),
        sequence_question_words=bool(re.search(r'\b(before|after|first|last|then|when|while|next)\b',text)),
        source='question text only, not video content or model output'))
save(OUT/'question_proxies_LOCAL_ONLY.json',dict(rows=proxies,
    note='overlapping question-word proxies only; not verified static detail, motion or small-object labels'))
save(OUT/'analysis_lock.json',dict(scorer_SHA256=sha(HERE/'analyze.py'),
    labels_SHA256=sha(OUT/'labels_LOCAL_ONLY.json'),references_SHA256=sha(OUT/'historical_references_LOCAL_ONLY.json'),
    proxies_SHA256=sha(OUT/'question_proxies_LOCAL_ONLY.json')))
print(dict(files=len(lock),execution_lock_sha256=sha(run/'execution_lock.json'),GPU_authorized=False))
