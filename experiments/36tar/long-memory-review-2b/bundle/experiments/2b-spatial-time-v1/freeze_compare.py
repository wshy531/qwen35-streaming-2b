"""Freeze one capacity-controlled candidate without GPU authorization."""
import ast
import gzip
import json
from pathlib import Path
from evaluate_compare import ROOT,read,save,sha
from review_media import OUT


def main():
    run=OUT/'runtime_inputs_v2';run.mkdir(exist_ok=False)
    previous=ROOT/'runs/2b_long_stream_prepare_20261020_01/runtime_inputs_v2'
    for name in ('sources.json','requests.json','snapshots.json'):save(run/name,read(previous/name))
    lock=read(ROOT/'runs/2b_memory_readability_prepare_20261020_01/runtime_inputs/runtime_lock.json')
    for name,h in lock.items():
        p=ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==h,name
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.sh'):
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
            lock[p.relative_to(ROOT).as_posix()]=sha(p)
    proof=read(OUT/'CPU_policy1426.json');assert proof['status']=='passed'
    # The mechanism proof predates the worker: bind only files actually executed then.
    for name in ('policy.py','cpu_contract.py'):
        assert sha(Path(__file__).parent/name)==proof['source_sha256'][name]
    with (run/'fixtures.json.gz').open('xb') as f:
        f.write(gzip.compress(json.dumps(proof['fixtures'],separators=(',',':')).encode(),mtime=0))
    save(run/'runtime_lock.json',lock)
    save(run/'CPU_policy_proof.json',dict(status='passed',sha256=sha(OUT/'CPU_policy1426.json'),
        pairs=proof['pairs'],executed_policy_sha256={n:proof['source_sha256'][n] for n in ('policy.py','cpu_contract.py')},
        features='synthetic32; not real2B capability evidence'))
    protocol=dict(status='fixed_A04_capacity_and_single_S24_development_comparison',
        arms=['M0_384','M2_384','M0_768','M2_768','S24_384'],sources=6,questions=34,
        original28_preserved=True,supported_original=16,manual_delayed=6,original_ambiguous=11,
        original_not_established=1,all_are_open_development=True,
        hypothesis='pair-local spatial reduction may retain more separate old time nodes at equal384 visual tokens',
        allocation=dict(M0_384='recent8x48',M2_384='recent4x48+open48+closed144',
            M0_768='recent16x48',M2_768='recent4x48+open48+closed528',
            S24_384='recent4x48+open24+closed168'),
        S24='existing ToMe48to24 within one promoted raw pair; original surviving xy/time; temporal merging remains',
        unchanged=['causal event flags','original closed cost','merge formula','reader','question text','FP32 weights'],
        sampling_fps=1,canvas_hw=[192,256],pair_rule='complete real pairs only, no query padding',
        shared_causal_vision=True,shared_main_context_tokens=2048,
        answers_max=170,generated_tokens_max=1360,vision_calls_max=1426,pretrained_loads_max=1,
        partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,host_memory_GiB=24,
        wall_seconds=1800,worker_seconds=1500,process_RSS_guard_bytes=20*2**30,
        GPU_allocated_guard_bytes=32*2**30,log_cap_bytes=96*2**20,
        no_training=True,no_download=True,no_auto_retry=True,no_full_history_feature_archive=True,
        causal_stream_with_midstream_token_callbacks=True,wallclock_1x=False,
        model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
        analysis='strict frozen parser; all28/supported16/manual6; capacity four first then S24 vs384; corrections/losses; M0/M2 original IDs reproducibility',
        stop='one fixed comparison then analyze; no new budgets/spatial ratios/prompts; larger heldout data required before capability claim',
        caveats=['six reused sources not formal20/160','S24 may damage small spatial detail',
            '768 uses twice storage and is capacity control, not algorithm gain',
            'shared process timings not per-arm speed','native217776 reference not rerun'],
        input_sha256={p.name:sha(p) for p in run.iterdir()})
    save(run/'protocol.json',protocol)
    print(dict(protocol_SHA256=sha(run/'protocol.json'),runtime_files=len(lock),GPU_authorized=False))


if __name__=='__main__':main()
