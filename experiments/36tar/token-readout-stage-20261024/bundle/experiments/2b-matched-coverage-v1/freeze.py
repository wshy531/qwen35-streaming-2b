"""Freeze exact source/input/CPU receipts, with no GPU authorization."""
import ast
from pathlib import Path
from common import ROOT,OUT,ARMS,read,save,sha


def main():
    here=Path(__file__).parent;run=OUT/'inputs'
    cpu=read(OUT/'CPU_preflight.json');tiny=read(OUT/'tiny_session_check.json')
    assert cpu['status']==tiny['status']=='passed' and cpu['layouts']==1515
    assert cpu['legacy_step_regressions']==288 and len(tiny['QA'])==5
    assert tiny['post_QA_real_pair_continuation']
    source=ROOT/'runs/2b_compressed_banks_prepare_20261023_01'
    for name in ('groups.json','questions.json','snapshots34.json'):
        assert sha(run/name)==sha(source/'inputs'/name)
    assert sha(OUT/'labels_LOCAL_ONLY.json')==sha(source/'labels_LOCAL_ONLY.json')
    lock=read(source/'inputs/execution_lock.json')['runtime_sha256'].copy()
    for rel,digest in lock.items():
        path=ROOT/rel if not rel.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for path in here.iterdir():
        if path.suffix in ('.py','.sh'):
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
            lock[path.relative_to(ROOT).as_posix()]=sha(path)
    save(run/'execution_lock.json',dict(protocol_sha256=sha(run/'protocol.json'),
        input_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},runtime_sha256=lock,
        local_CPU_sha256=sha(OUT/'CPU_preflight.json'),local_tiny_sha256=sha(OUT/'tiny_session_check.json'),
        no_GPU_authorization=True,declared_GPU_proposal=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,
        host_memory_GiB=24,wall_seconds=2700,worker_seconds=2400,answers_max=1515,
        generated_tokens_max=12120,visual_encodings=3981,pretrained_loads=1,log_bytes_cap=64*2**20,
        no_auto_retry=True,no_training=True,no_downloads=True)))
    assert len(read(OUT/'historical_references_LOCAL_ONLY.json')['records'])==909
    save(OUT/'analysis_lock.json',dict(scorer_SHA256=sha(here/'analyze.py'),labels_SHA256=sha(OUT/'labels_LOCAL_ONLY.json'),
        historical_references_SHA256=sha(OUT/'historical_references_LOCAL_ONLY.json')))
    print(dict(status='frozen',source_files=len(lock),lock_SHA256=sha(run/'execution_lock.json'),GPU_authorized=False))


if __name__=='__main__':main()
