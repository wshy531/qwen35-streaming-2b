"""Freeze only after real CPU integration; GPU authorization remains absent."""
import ast
from pathlib import Path
from common import ROOT,OUT,read,save,sha

cpu=read(OUT/'CPU_preflight.json');mechanism=read(OUT/'CPU_mechanism_v3.json')
current=sha(Path(__file__).parent/'cacher.py')
assert cpu['status']==mechanism['status']=='passed' and cpu['layouts']==606
assert cpu['cacher_SHA256']==mechanism['source_SHA256']==current
old=ROOT/'runs/2b_standalone_cost_prepare_20261023_01/inputs/execution_lock.json'
lock=read(old)['runtime_sha256'].copy()
for name,digest in lock.items():
    file=ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
    assert sha(file)==digest,name
for file in Path(__file__).parent.iterdir():
    if file.suffix in ('.py','.sh'):
        if file.suffix=='.py':ast.parse(file.read_text(encoding='utf-8'))
        lock[file.relative_to(ROOT).as_posix()]=sha(file)
save(OUT/'inputs/execution_lock.json',dict(runtime_SHA256=lock,
    input_SHA256={f.name:sha(f) for f in (OUT/'inputs').iterdir() if f.is_file()},
    local_CPU_SHA256=sha(OUT/'CPU_preflight.json'),local_mechanism_SHA256=sha(OUT/'CPU_mechanism_v3.json'),
    GPU_authorization=False,proposal=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,
        RAM_GiB=24,minutes=45,answers_max=608,generated_tokens_max=4864,visual_calls_max=7982,
        pretrained_loads=1,log_bytes_max=64*2**20,no_retry=True,no_training=True,no_downloads=True)))
save(OUT/'analysis_lock.json',dict(scorer_SHA256=sha(Path(__file__).parent/'analyze.py'),
    labels_SHA256=sha(OUT/'labels_LOCAL_ONLY.json'),
    historical_references_SHA256=sha(OUT/'historical_references_LOCAL_ONLY.json')))
print(dict(status='frozen',runtime_files=len(lock),GPU_authorization=False))
