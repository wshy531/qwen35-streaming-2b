"""Freeze E2 only after legacy/ephemeral layout and real tiny-QA checks."""
import ast
from pathlib import Path
from common import ROOT,OUT,read,save,sha
cpu=read(OUT/'CPU_preflight.json');tiny=read(OUT/'tiny_session_check.json')
assert cpu['status']==tiny['status']=='passed' and cpu['layouts']==1818 and len(tiny['QA'])==6
assert cpu['legacy_step_regressions']==384 and cpu['M0_equivalence_checks']==128
lock=read(ROOT/'runs/2b_compressed_banks_prepare_20261023_01/inputs/execution_lock.json')['runtime_sha256'].copy()
for rel,digest in lock.items():
    p=ROOT/rel if not rel.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
    assert sha(p)==digest,rel
for p in Path(__file__).parent.iterdir():
    if p.suffix in ('.py','.sh'):
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        lock[p.relative_to(ROOT).as_posix()]=sha(p)
run=OUT/'inputs'
save(run/'execution_lock.json',dict(protocol_sha256=sha(run/'protocol.json'),
    input_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},runtime_sha256=lock,
    local_CPU_sha256=sha(OUT/'CPU_preflight.json'),local_tiny_sha256=sha(OUT/'tiny_session_check.json'),
    authorization_basis='user approved E1/E2 and subsequent PR on20261024; cluster CPU checks required'))
save(OUT/'analysis_lock.json',dict(scorer=sha(Path(__file__).with_name('analyze.py')),
    audit_scorer=sha(Path(__file__).with_name('audit_and_decide.py')),
    labels=sha(OUT/'labels_LOCAL_ONLY.json'),references=sha(OUT/'historical_references_LOCAL_ONLY.json')))
print(dict(status='frozen',runtime_files=len(lock),lock_sha256=sha(run/'execution_lock.json')))
