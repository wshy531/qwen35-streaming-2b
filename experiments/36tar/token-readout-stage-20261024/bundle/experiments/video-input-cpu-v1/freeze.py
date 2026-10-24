"""Freeze the complete CPU contrast before any timed case."""
import ast
from common import ROOT,OUT,HERE,read,save,sha

assert read(OUT/'CPU_preflight.json')['status']=='passed'
files=list(HERE.glob('*.py'))+[ROOT/'experiments/2b-natural-eval-v1/audit_sampling.py',
    ROOT/'experiments/2b-natural-eval-v1/memory_policy.py',ROOT/'experiments/2b-long-stream-v1/schedule.py']
for file in files:ast.parse(file.read_text(encoding='utf-8'))
save(OUT/'inputs/execution_lock.json',dict(
    runtime_SHA256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
    input_SHA256={p.name:sha(p) for p in (OUT/'inputs').iterdir() if p.is_file()},
    CPU_preflight_SHA256=sha(OUT/'CPU_preflight.json'),GPU_jobs=0))
print(dict(status='frozen',sources=len(files)),flush=True)
