"""Answer-blind single A06 first-version protocol; no GPU authorization."""
import ast
from pathlib import Path

from policy import ROOT
from audit_nodes import OUT,OLD
from evaluate_compare import read,save,sha


def main():
    run=OUT/'runtime_inputs';run.mkdir(exist_ok=False)
    previous=ROOT/'runs/2b_long_stream_prepare_20261020_01/runtime_inputs_v2'
    for name in ('sources.json','requests.json','snapshots.json'):save(run/name,read(previous/name))
    lock=read(OLD/'runtime_inputs/runtime_lock.json')
    for name,h in lock.items():
        path=ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(path)==h,name
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh'):
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
            lock[path.relative_to(ROOT).as_posix()]=sha(path)
    save(run/'runtime_lock.json',lock)
    proof=read(OUT/'CPU_contract1426.json');assert proof['status']=='passed'
    save(run/'CPU_policy_proof.json',dict(status='passed',sha256=sha(OUT/'CPU_contract1426.json'),
        pairs=proof['pairs'],source_sha256=proof['source_sha256'],features='synthetic32; not capability evidence'))
    protocol=dict(status='fixed_A06_first_version_development_comparison',arms=['M0','M2','A06'],
        sources=6,questions=34,original28_preserved=True,supported_original=16,manual_delayed=6,
        original_ambiguous=11,original_not_established=1,all_are_open_development=True,
        hypothesis='closed-root depth priority may reduce repeated merge damage under identical384 payload budget',
        selector='min(parent depth, frozen original penalty total, adjacent index)',
        unchanged=['four recent pairs','open48','closed144','merge formula','causal event flags','positions','reader','questions','FP32 weights'],
        sampling_fps=1,canvas_hw=[192,256],pair_rule='complete real pairs only, no query padding',
        shared_causal_vision=True,shared_main_context_tokens=2048,bank_visual_tokens=384,
        answers_max=102,generated_tokens_max=816,vision_calls_max=1426,pretrained_loads_max=1,
        partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,host_memory_GiB=24,wall_seconds=1800,worker_seconds=1500,
        process_RSS_guard_bytes=20*2**30,GPU_allocated_guard_bytes=32*2**30,log_cap_bytes=64*2**20,
        no_training=True,no_download=True,no_auto_retry=True,no_full_history_feature_archive=True,
        causal_stream_with_midstream_token_callbacks=True,wallclock_1x=False,
        model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
        analysis='strict original parser, supported16/manual6/all28 separate; paired corrections/losses, old M0/M2 generated-ID replay; no posthoc rescore',
        stop='one fixed candidate comparison then analyze; no ordering/weight/prompt sweep',
        caveats=['six reused sources, not formal20/160','shared process timings are not independent arm speed',
            'depth reduction is not itself ability or information preservation','full native217776 retained as previous reference, not rerun'],
        input_sha256={p.name:sha(p) for p in run.iterdir()})
    save(run/'protocol.json',protocol);print(dict(protocol_SHA256=sha(run/'protocol.json'),runtime_files=len(lock),GPU_authorized=False))


if __name__=='__main__':main()
