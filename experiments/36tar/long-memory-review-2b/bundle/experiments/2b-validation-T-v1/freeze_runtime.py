"""Freeze worker-safe inputs and exact existing dependencies; no submission."""
import ast
from pathlib import Path

from prepare import ROOT, OUT, read, save, sha


def main():
    destination = OUT/'runtime_inputs'
    destination.mkdir(exist_ok=False)
    requests = read(OUT/'requests.json')
    media = {f['media_path']:f for f in read(OUT/'download_manifest.json')['files']}
    for row in requests:
        original = media[row['media_path']]
        row['media_path'] = 'data/T-new40-20261020/'+original['source_path']+'.mp4'
    save(destination/'requests.json',requests)
    previous = ROOT/'runs/2b_short_event_prepare_20261020_01'
    locks = {}
    for rel,digest in read(previous/'dependency_lock.json').items():
        path = ROOT/'experiments/2b-natural-eval-v1'/rel
        assert sha(path)==digest
        locks[path.relative_to(ROOT).as_posix()] = digest
    policy = ROOT/'experiments/2b-short-event-v1/policies.py'
    assert sha(policy)==read(previous/'source_lock.json')['policies.py']
    locks[policy.relative_to(ROOT).as_posix()] = sha(policy)
    for name in ['evaluate.py','run_eval.sh']:
        path = Path(__file__).parent/name
        locks[path.relative_to(ROOT).as_posix()] = sha(path)
        if path.suffix=='.py':
            ast.parse(path.read_text(encoding='utf-8'))
    save(destination/'runtime_lock.json',locks)
    save(destination/'runtime_protocol.json',dict(
        status='fixed_T_comparison_CPU_preflight_pending_GPU_NOT_authorized',questions=98,sources=39,
        arms=dict(native=['native_original'],memory=['full_pairwise_reference','M0','M1','M2',
                                                    'B02_cap4','reservoir4','text_only']),
        maximum_answers=784,maximum_generated_tokens=6272,max_new_tokens=8,maximum_model_loads=2,
        common_full_context_tokens=8192,common_full_visual_tokens=6144,candidate_visual_tokens=384,
        original_native_guard3072_replaced_by_explicit8192_in_new_adapter_only=True,
        no_truncation=True,sampling_fps=1,canvas_hw=[192,256],precision='float32',
        fixed_threshold=.047536224126815796,B02_cap_pairs=4,reservoir_seed=20261020,
        worker_seconds=1500,slurm_minutes=60,gpus=1,cpus=4,memory_GiB=24,
        partition='g078t',comment='gx_llm',automatic_retries=0,
        requests_sha256=sha(destination/'requests.json'),runtime_lock_sha256=sha(destination/'runtime_lock.json'),
        local_qualification_sha256=sha(OUT/'content_review_receipt.json'),
        local_analysis_protocol_sha256=sha(OUT/'analysis_protocol.json'),
        no_labels_or_intervals_in_worker=True,
        stop='one fixed new-source comparison, no parameter/prompt search',
        limits='single-assistant review, one180s C-only source, indirect late cues; not proven old-evidence necessity'))
    print(dict(worker_inputs=98,dependency_files=len(locks),maximum_answers=784,GPU_authorized=False))


if __name__=='__main__':
    main()
