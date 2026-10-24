"""Freeze answer-blind runtime inputs; no remote operations."""
import ast
from pathlib import Path
from prepare import ROOT,OUT,OLD,read,sha
from evaluate import save


def main():
    here=Path(__file__).parent;run=OUT/'runtime_inputs';run.mkdir(exist_ok=False)
    rows=read(OUT/'requests6_LOCAL.json')
    for row in rows:del row['local_media_path']
    save(run/'requests.json',rows)
    save(run/'plan.json',read(OUT/'diagnostic_plan.json'))
    lock=read(OLD/'runtime_inputs_v2/runtime_lock.json')
    for name,h in lock.items():
        p=ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==h,name
    for p in here.iterdir():
        if p.suffix in ('.py','.sh'):
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
            lock[p.relative_to(ROOT).as_posix()]=sha(p)
    save(run/'runtime_lock.json',lock)
    proofs=['CPU_clock_contracts.json','CPU_clock_receipt.json','CPU_diagnostic_contexts36.json']
    save(run/'local_CPU_proofs.json',{n:dict(sha256=sha(OUT/n),proof=read(OUT/n)) for n in proofs})
    protocol=dict(status='preregistered_single_diagnostic_NOT_architecture_improvement',
        cases=6,answers_max=36,primary_ABCD=24,reference_EF=12,generated_tokens_max=288,
        question_labels_unchanged=True,node_selection='oracle witnesses fixed before previous GPU output; not automatic retrieval',
        source_ranges_A_B_equal=True,recent4_C_D_bitwise_equal=True,
        compressed_positions='existing surviving-coordinate heuristic unchanged; A/B cannot isolate features from positions',
        reference_F_required_to_reproduce_old_generated_IDs=True,
        reference_E_required_to_separate_pairwise_from_native_readout=True,
        partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,host_memory_GiB=24,wall_seconds=1800,
        worker_seconds=1500,pretrained_loads_max=1,precision='FP32',model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
        vision_calls_max=sum(r['total_prefix_pairs'] for r in rows),
        context_guard=32768,visual_guard=16384,nodes_guard=256,prefill_chunk=512,
        snapshots_total_cap_bytes=24*2**20,process_RSS_guard_bytes=20*2**30,GPU_allocated_guard_bytes=32*2**30,
        raw_CPU_features='temporary full arrived prefix per-source only, diagnostic not bounded candidate',
        no_training=True,no_download=True,no_camera=True,no_automatic_retry=True,
        causal_replay=True,real_1x_GPU_test=False,default_unchanged=True,
        stop='one fixed36-answer round then classify; no prompt/threshold/node sweeps',
        inputs_sha256={p.name:sha(p) for p in run.iterdir()})
    save(run/'protocol.json',protocol)
    print(dict(runtime_files=len(lock),protocol_SHA256=sha(run/'protocol.json'),GPU_authorized=False))


if __name__=='__main__':main()
