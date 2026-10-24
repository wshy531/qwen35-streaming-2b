"""Bind exact code and CPU-checked inputs without granting GPU authorization."""
import ast
from pathlib import Path
import shutil
from common import ROOT, OUT, read, save, sha


def main():
    run=OUT/'inputs'
    cpu=read(OUT/'CPU_preflight.json')
    assert cpu['status']=='passed' and cpu['protocol_sha256']==sha(run/'protocol.json')
    previous=ROOT/'runs/2b_memory_readability_prepare_20261020_01/runtime_inputs/runtime_lock.json'
    lock=read(previous)
    for relative,digest in lock.items():
        path=ROOT/relative if not relative.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/relative[5:]
        assert sha(path)==digest,relative
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh'):
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
            lock[path.relative_to(ROOT).as_posix()]=sha(path)
    shutil.copyfile(ROOT/'runs/2b_long_stream_prepare_20261020_01/runtime_inputs_v2/snapshots.json',run/'snapshots34.json')
    save(run/'execution_lock.json',dict(protocol_sha256=sha(run/'protocol.json'),
        input_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},runtime_sha256=lock,
        local_CPU_sha256=sha(OUT/'CPU_preflight.json'),no_GPU_authorization=True,
        declared_GPU_proposal=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,host_memory_GiB=24,
            wall_seconds=3600,worker_seconds=3300,pretrained_loads=1,answers_max=1515,generated_tokens_max=12120,
            visual_encodings=3981,log_bytes_cap=64*2**20,no_training=True,no_download=True,no_auto_retry=True)))
    references=[]
    for relative in ('runs/2b_natural_eval_20261020_01/gpu_217141/result/memory/predictions.jsonl',
                     'runs/2b_validation_T_prepare_20261020_01/gpu_217215/result/memory/predictions.jsonl'):
        import json
        for line in (ROOT/relative).open(encoding='utf-8'):
            item=json.loads(line)
            if item['arm']=='full_pairwise_reference':
                references.append(dict(id=item['id'],generated_ids=item['generated_ids'],
                    source=relative,source_sha256=sha(ROOT/relative)))
    assert len(references)==len({r['id'] for r in references})==269
    save(OUT/'raw48_references_LOCAL_ONLY.json',references)
    save(OUT/'analysis_lock.json',dict(analysis_sha256=sha(Path(__file__).parent/'analyze.py'),
        labels_sha256=sha(OUT/'labels_LOCAL_ONLY.json'),input_provenance_sha256=sha(OUT/'input_provenance.json'),
        raw48_references_sha256=sha(OUT/'raw48_references_LOCAL_ONLY.json')))
    print(dict(runtime_files=len(lock),execution_lock_sha256=sha(run/'execution_lock.json'),GPU_authorized=False))


if __name__=='__main__':main()
