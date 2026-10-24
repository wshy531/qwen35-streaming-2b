"""Bind exact workload, old source dependencies and both local CPU receipts."""
import ast
from pathlib import Path
from common import ROOT,OUT,read,save,sha


def main():
    run=OUT/'inputs';here=Path(__file__).parent
    cpu=read(OUT/'CPU_preflight.json');tiny=read(OUT/'tiny_session_check.json')
    assert cpu['status']==tiny['status']=='passed' and cpu['layouts']==909
    assert cpu['legacy_step_regressions']==288 and len(tiny['QA'])==3
    source=ROOT/'runs/2b_matched_coverage_prepare_20261023_01'
    for name in ('groups.json','questions.json','snapshots34.json'):
        assert sha(run/name)==sha(source/'inputs'/name)
    lock=read(source/'inputs/execution_lock.json')['runtime_sha256'].copy()
    for rel,digest in lock.items():
        path=ROOT/rel if not rel.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for p in here.iterdir():
        if p.suffix in ('.py','.sh'):
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
            lock[p.relative_to(ROOT).as_posix()]=sha(p)
    save(run/'execution_lock.json',dict(protocol_sha256=sha(run/'protocol.json'),
        input_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},runtime_sha256=lock,
        local_CPU_sha256=sha(OUT/'CPU_preflight.json'),local_tiny_sha256=sha(OUT/'tiny_session_check.json'),
        no_GPU_authorization=True,declared_GPU_proposal=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,
        host_memory_GiB=24,wall_seconds=4500,worker_seconds=4200,answers_max=918,measured_answers=909,
        generated_tokens_max=7344,visual_encodings=12015,pretrained_loads=9,log_bytes_cap=64*2**20,
        no_auto_retry=True,no_training=True,no_downloads=True)))
    save(OUT/'analysis_lock.json',dict(quality_scorer_SHA256=sha(here/'analyze.py'),cost_scorer_SHA256=sha(here/'analyze_cost.py'),
        labels_SHA256=sha(OUT/'labels_LOCAL_ONLY.json'),historical_references_SHA256=sha(OUT/'historical_references_LOCAL_ONLY.json')))
    print(dict(status='frozen',runtime_files=len(lock),lock_SHA256=sha(run/'execution_lock.json'),GPU_authorized=False))


if __name__=='__main__':main()
