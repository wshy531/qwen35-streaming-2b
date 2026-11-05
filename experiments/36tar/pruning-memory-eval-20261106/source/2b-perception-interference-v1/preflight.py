"""CPU-only causal layout, intervention, scoring and random mixed-model checks."""
import argparse
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import torch
from common import ROOT, OLD, INPUTS, PREP, ARMS, read, save, sha, displayed, dependencies, require_admission
from interference_readout import compose, split, swapped_history
from scoring import scored, summarize
from sember_common import runtime
from sember_banks import Bank
from sember_nodes import pair_node, penalties, merge, digest
from sember_runtime import make_builder
from sember_protocol import prompt


def tensor_sha(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def expect_error(function, exception=ValueError):
    try:
        function()
    except exception:
        return
    raise AssertionError('invalid condition was accepted')


class EmbeddingsOnly:
    def __init__(self):
        self.embedding = torch.nn.Embedding(248320, 32)

    def get_input_embeddings(self):
        return self.embedding


def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CPU-only preflight requires CUDA_VISIBLE_DEVICES=-1')
    torch.set_num_threads(1)
    torch.manual_seed(20261027)
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    tasks = read(INPUTS / 'tasks_LOCAL_ONLY.json')
    schedule = read(INPUTS / 'schedule.json')
    assert len(tasks) == 6 and len(schedule) == 90
    assert sum(r['content_scored'] for r in schedule) == 75
    assert len({(r['task_id'], r['arm'], r['shift']) for r in schedule}) == 90
    for path, expected in manifest['input_origin_sha256'].items():
        assert sha(ROOT / path) == expected
    for pair in manifest['pairs']:
        assert sha(ROOT / pair['image_relative_path']) == pair['image_sha256']
        assert pair['committed_through_s'] <= pair['query_time_s']
        assert pair['donor_source_path'] != pair['source_path']
        assert pair['donor_prefix_last_s'] < pair['committed_through_s']
    reviews = read(PREP / 'independent_review_LOCAL_ONLY.json')
    expect_error(lambda: require_admission(manifest, reviews))

    # Real CPU reference: compare our bank scheduling and all tensors to the
    # partner's byte-pinned scheduling snapshot, injecting the same frozen math.
    source = OLD / 'partner_readonly/policy.py'
    receipt = read(OLD / 'source_execution_binding.json')['read_source_matches_GPU_lock']
    assert sha(source) == receipt['experiments/wshy/perception_memory_2b/policy.py']
    spec = importlib.util.spec_from_file_location('interference_partner_partition', source)
    partner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = partner
    spec.loader.exec_module(partner)
    bank = Bank('LS', 108)
    control = partner.Partition(partner.Allocation(4, 108), penalties, merge)
    parity_steps = 64
    for index in range(parity_steps):
        value = torch.randn(108, 32)
        node = pair_node(value, index, float(2 * index), float(2 * index + 1), (9, 12))
        starts = index % 5 == 0 or index % 13 == 0
        bank.append(node, starts_event=starts)
        control.append(node, starts_event=starts)
        assert digest(bank.nodes()) == digest(control.nodes())
        assert bank.within_merges == control.within and bank.long_merges == control.cross

    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(ROOT / 'models/Qwen3.5-2B', local_files_only=True)
    builder = make_builder(env, processor)
    model = EmbeddingsOnly()
    donors = [torch.randn(108, 32) for _ in range(4)]
    before = digest(bank.nodes())
    history, recent = split(bank)
    changed = swapped_history(history, donors)
    assert digest(bank.nodes()) == before
    for a, b in zip(history, changed):
        assert a.metadata() == b.metadata()
        assert torch.equal(a.mass, b.mass) and torch.equal(a.coords, b.coords)
        assert not torch.equal(a.features, b.features)
    changed[0].features.zero_()
    assert digest(bank.nodes()) == before
    expect_error(lambda: swapped_history(history, donors[:3]))
    expect_error(lambda: swapped_history(history, [torch.randn(107, 32)] * 4))
    layouts = {}
    maximum = 0
    for task in tasks:
        for shift in range(5):
            q, mapping = displayed(task, shift)
            assert mapping.index(task['correct_index']) == (task['correct_index'] - shift) % 5
            text = prompt(q)
            poisoned = dict(task, correct_index=(task['correct_index'] + 1) % 5,
                            independent_review='ANSWER_KEY_MUST_NOT_LEAK')
            assert prompt(displayed(poisoned, shift)[0]) == text
            versions = {arm: compose(model, builder, bank, text, arm, donors) for arm in ARMS}
            full, selected, swap = versions['full'], versions['recent'], versions['history_value_swap']
            removed = full[2].shape[1] - selected[2].shape[1]
            prefix = builder.prefix_ids.shape[1]
            assert torch.equal(full[0][:, prefix + removed:], selected[0][:, prefix:])
            assert torch.equal(full[1][1:, :, prefix + removed:], selected[1][1:, :, prefix:])
            assert torch.equal(full[2], swap[2]) and torch.equal(full[1], swap[1])
            assert not torch.equal(full[0], swap[0])
            for arm, (emb, pos, ids, meta) in versions.items():
                assert emb.shape[1] == ids.shape[1] == pos.shape[-1]
                assert torch.equal(pos[0, 0], torch.arange(ids.shape[1]))
                assert meta['visual_tokens'] == (432 if arm == 'recent' else 864)
                assert ids.shape[1] + 8 <= 8192 and meta['suffix_tokens'] <= 512
                key = f"{task['id']}|{arm}|{shift}"
                layouts[key] = dict(input_ids_sha256=tensor_sha(ids), positions_sha256=tensor_sha(pos),
                    input_tokens=meta['input_tokens'], visual_tokens=meta['visual_tokens'])
                maximum = max(maximum, ids.shape[1])
    assert len(layouts) == 90 and digest(bank.nodes()) == before

    # Scorer regression includes genuine content-correct text with extra words,
    # option remapping, missing predictions and artificial old-evidence replacement.
    fake = []
    tasks_byid = {t['id']: t for t in tasks}
    for row in schedule:
        task = tasks_byid[row['task_id']]
        letter = chr(65 + (task['correct_index'] - row['shift']) % 5)
        result = scored(task, row['arm'], row['shift'], f'Answer: {letter}. This is the visible item.')
        assert result['format_only'] is False
        assert result['content_correct'] is (True if row['content_scored'] else None)
        fake.append(result)
    regression = summarize(fake, tasks, schedule)
    assert sum(x['denominator'] for x in regression['totals'].values()) == 75
    assert regression['both_questions_correct'] == {'recent': 15, 'full': 15}
    expect_error(lambda: summarize(fake[:-1], tasks, schedule))
    expect_error(lambda: summarize(fake + fake[:1], tasks, schedule))
    assert scored(tasks[0], 'recent', 0, 'unknown')['content_correct'] is False
    assert scored(tasks[0], 'history_value_swap', 0, 'unknown')['content_correct'] is None

    # Exercise fresh mixed KV/conv/recurrent QA with random small weights.
    tiny = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**env['tiny_config'](processor), attn_implementation='sdpa')).float().eval()
    width = tiny.get_input_embeddings().embedding_dim
    micro_bank = Bank('LS', 108)
    for index in range(10):
        micro_bank.append(pair_node(torch.randn(108, width), index, float(index * 2),
                                   float(index * 2 + 1), (9, 12)), starts_event=index % 3 == 0)
    micro_donors = [torch.randn(108, width) for _ in range(4)]
    micro_before = digest(micro_bank.nodes())
    micro_rows = []
    saved = tiny.model.rope_deltas
    with torch.inference_mode(), env['fp32_reference']():
        for arm in (*ARMS, 'recent'):
            emb, pos, _, _ = compose(tiny, builder, micro_bank, 'What is visible now?', arm, micro_donors)
            sequence = env['HybridSequence'](tiny, max_tokens=8192, session_id='CPU-interference-' + arm)
            try:
                logits = None
                for start in range(0, emb.shape[1], 512):
                    end = min(start + 512, emb.shape[1])
                    logits = sequence.append(inputs_embeds=emb[:, start:end], position_ids=pos[:, :, start:end],
                        logits_mode='last' if end == emb.shape[1] else 'none')
                assert torch.isfinite(logits).all()
                token = int(logits[0, -1].argmax())
                logits = sequence.append(input_ids=torch.tensor([[token]]), logits_mode='last')
                assert torch.isfinite(logits).all()
                micro_rows.append(dict(arm=arm, first_token=token, seen_tokens=sequence.seen_tokens))
            finally:
                sequence.cache.reset()
                tiny.model.rope_deltas = saved
            assert digest(micro_bank.nodes()) == micro_before
    assert micro_rows[0] == micro_rows[-1], 'fresh repeated current read drifted'
    proof = dict(status='passed_CPU_only_not_admitted', planned_layouts=len(layouts), layouts=layouts,
        maximum_input_tokens=maximum, partner_schedule_tensor_parity_steps=parity_steps,
        full_swap_same_ids_and_positions=True, recent_suffix_embedding_and_mrope_exact=True,
        intervention_changes_only_historical_visual_values=True, answer_keys_not_rendered=True,
        bank_digest_unchanged=True, scorer_regression=regression, random_tiny_mixed_checks=micro_rows,
        fixture='synthetic 64-pair bank, not per-query real visual history',
        GPU_jobs=0, pretrained_loads=0, random_tiny_models=1,
        independent_content_review='pending', capability_or_speed_claim=False,
        runtime_sha256=dependencies(),
        input_sha256={p.name: sha(p) for p in INPUTS.iterdir() if p.is_file()})
    save(output, proof)
    print({k: v for k, v in proof.items() if k not in ('layouts', 'scorer_regression', 'runtime_sha256', 'input_sha256')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
