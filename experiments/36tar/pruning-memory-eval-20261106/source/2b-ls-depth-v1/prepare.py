"""Reuse frozen question-time inputs without filtering by model outcomes."""
from pathlib import Path
import shutil
from common import ROOT, OUT, ARMS, read, save, sha


def main():
    previous = ROOT/'runs/2b_token_compression_prepare_20261021_01'
    frozen = read(previous/'inputs/execution_lock.json')
    for name in ('groups.json', 'questions.json', 'snapshots34.json'):
        assert sha(previous/'inputs'/name) == frozen['input_sha256'][name]
    OUT.mkdir(exist_ok=False)
    run = OUT/'inputs'; run.mkdir()
    for name in ('groups.json', 'questions.json', 'snapshots34.json'):
        shutil.copyfile(previous/'inputs'/name, run/name)
    for name in ('labels_LOCAL_ONLY.json', 'input_provenance.json'):
        shutil.copyfile(previous/name, OUT/name)
    snapshot = ROOT/'runs/memory_token_next_stage_20261021_01/shared_4212_snapshot.py'
    blob = hashlib_git(snapshot.read_bytes())
    assert blob == 'a1d39e23c0d0feb3cca706a642e336903365ae20'
    shutil.copyfile(snapshot, Path(__file__).parent/'reference_shared_4212.py')
    save(run/'protocol.json', dict(arms=list(ARMS), questions=303, primary_development=269,
        separate_stream_regression=34, groups=112, answers_max=1818, generated_tokens_max=14544,
        vision_encodings=3981, model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
        dtype='FP32', canvas_hw=[192,256], max_new_tokens=8, question_suffix_cap=128,
        context_cap=2048, per_bank_visual_cap=384, recent_tokens=192, LS_shared_history_tokens=192,
        hypothesis='Does depth-first adjacent overflow selection improve LS shared-history retention?',
        changed_variable='LS overflow selector only: depth then frozen cost then index',
        frozen_LS_commit='4212d2399954f19700b568cc806ed82fece3349f', frozen_LS_blob=blob,
        write_uses_question=False, full_history_archive=False, persistent_language_main=False,
        event_threshold=0.047536224126815796, reservoir_seed=20261020,
        short_odd_tail='query-only duplicate last real frame; session may not continue',
        long_odd_tail='pending real frame never padded at query; continue to EOF',
        bank_readout='existing readout.context, original destination source coordinates; heuristic compression positions',
        historical_costs_not_comparable='bank-only session has no observation main language state',
        scoring='answer correctness primary; format separate; ambiguous/missing keep denominator',
        known_limits=['opened development data', 'not independent filming groups',
            'long28 has16supported11ambiguous1notestablished; manual6 single review',
            'independent second content review incomplete', 'not 1x wallclock'],
        stop='one fixed full comparison, no tuning or default promotion; analyze before Div24',
        resources=dict(partition='g078t', comment='gx_llm', GPUs=1, CPUs=4,
            host_memory_GiB=24, wall_seconds=3600, worker_seconds=3300, pretrained_loads=1,
            output_bytes_cap=96*2**20, no_training=True, no_download=True, no_auto_retry=True),
        input_sha256={p.name:sha(p) for p in run.iterdir()}))
    print(dict(prepared=True, questions=303, answers=1818, GPU_submissions=0))


def hashlib_git(data):
    import hashlib
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


if __name__ == '__main__':
    main()
