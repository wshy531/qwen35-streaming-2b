"""CPU-only contract and frozen303 layout checks; no capability scores."""
import os
import time
import importlib
from unittest.mock import patch
import torch
from common import ROOT, OUT, ARMS, runtime, read, save, sha
from banks import Banks
from probe import attention_rows, score_visual, select_half, question_rows
from bank_readout import context, Limits
from policy_adapters import BoundedShared
from features import raw_block
from online_bank import nodes_digest, load_distance, THRESHOLD


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1);torch.manual_seed(942)
    started = time.monotonic()
    OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
    source = ROOT/'runs/2b_readout_compression_prepare_20261024_01/inputs'
    for name in ('groups.json', 'questions.json', 'snapshots34.json'):
        (OUT/'inputs'/name).write_bytes((source/name).read_bytes())
    env = runtime()
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        ROOT/'models/Qwen3.5-2B', local_files_only=True), canvas=(192, 256), resize_mode='letterbox')
    cfg = env['tiny_config'](builder.processor)
    cfg['text_config'].update(num_hidden_layers=8, layer_types=(['linear_attention']*3+['full_attention'])*2,
                             num_attention_heads=4, num_key_value_heads=1)
    cfg['text_config']['rope_parameters']['partial_rotary_factor'] = .5
    cfg['text_config']['rope_parameters']['mrope_section'] = [1, 1, 2]
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **cfg, attn_implementation='sdpa')).float().eval()
    text = model.model.language_model;attn = text.layers[3].self_attn
    importlib.import_module(type(attn).__module__)
    checks = []
    with torch.inference_mode(), env['fp32_reference']():
        # Reference is the ACTUAL module eager full-attention result.
        hidden = torch.randn(1, 57, 32)
        positions = torch.stack([torch.arange(57), torch.arange(57)*2, torch.arange(57)//3])[:, None]
        rotary = text.rotary_emb(hidden, positions)
        mask = torch.full((1, 1, 57, 57), torch.finfo(torch.float32).min).triu(1)
        implementation = attn.config._attn_implementation
        try:
            attn.config._attn_implementation = 'eager'
            _, full = attn(hidden_states=hidden, position_embeddings=rotary, attention_mask=mask)
        finally:
            attn.config._attn_implementation = implementation
        rows = [2, 17, 56];reference = full[0, :, rows].mean((0, 1));worst = 0.
        for chunk in (1, 2, 16):
            scores, meta = attention_rows(attn, hidden, rotary, rows, row_chunk=chunk)
            worst = max(worst, (scores-reference).abs().max().item())
            torch.testing.assert_close(scores, reference, rtol=2e-6, atol=2e-7)
            assert meta['score_matrix_live_bytes_bound'] <= 2*4*min(chunk, 3)*57*4
        checks.append(dict(name='actual_eager_row_equivalence_partial_RoPE_GQA', worst_abs_error=worst))
        before, _ = attention_rows(attn, hidden, rotary, [2])
        future_changed = hidden.clone();future_changed[:, 3:] += 100
        after, _ = attention_rows(attn, future_changed, rotary, [2])
        assert torch.equal(before, after) and torch.count_nonzero(before[3:]) == 0
        checks.append(dict(name='future_key_mask', exact=True))

        bank, old = Banks(), BoundedShared(48)
        distance = load_distance();descriptor = None;selection_checks = 0
        for i in range(96):
            torch.manual_seed(200+i//5);features = torch.randn(48, 32)+i*.0001
            block = raw_block(features, i, i*2., i*2.+1);desc = features.mean(0)
            starts = descriptor is None or distance(descriptor, desc) > THRESHOLD
            old.append(block, starts_event=starts);descriptor = desc
            bank.append(features, i, i*2., i*2.+1);nodes = bank.full()
            assert nodes_digest(nodes) == nodes_digest(old.export())
            digest = bank.fingerprint();n = sum(len(x.features) for x in nodes)
            chosen, indices = select_half(nodes, torch.arange(n, dtype=torch.float32))
            assert sum(len(x.features) for x in chosen)*2 == n
            for a, b, keep in zip(nodes, chosen, indices):
                assert keep == list(range(len(a.features)//2, len(a.features)))
                assert torch.equal(b.coords, a.coords[keep]) and torch.equal(b.mass, a.mass[keep])
                assert b.features.data_ptr() != a.features.data_ptr()
                assert (a.first, a.last, a.begin, a.end, a.center, a.groups, a.depth) == (
                    b.first, b.last, b.begin, b.end, b.center, b.groups, b.depth)
                selection_checks += 1
            chosen[0].features.zero_();assert bank.fingerprint() == digest
            _, ties = select_half(nodes, torch.zeros(n))
            assert ties == [list(range(len(x.features)//2)) for x in nodes]
        checks.append(dict(name='legacy_LS_storage', steps=96, selected_node_checks=selection_checks))

        question = 'What appeared first? A. cat B. dog C. bird';nodes = bank.full()
        emb, pos, ids, meta = context(model, builder, nodes, question,
            limits=Limits(2048, 384, 16, 128), max_new_tokens=8)
        digest = bank.fingerprint();rows = question_rows(builder, question, ids.shape[1])
        suffix_start = ids.shape[1]-meta['suffix_tokens']
        assert min(rows) >= suffix_start and max(rows) < ids.shape[1]-1
        layers = []
        handles = [layer.register_forward_pre_hook(lambda m, a, k, i=i: layers.append(i), with_kwargs=True)
                   for i, layer in enumerate(text.layers)]
        hook_count = len(attn._forward_pre_hooks)
        try:
            scores, stats = score_visual(model, builder, question, emb, pos, ids)
        finally:
            for h in handles:h.remove()
        assert layers == [0, 1, 2, 3] and len(attn._forward_pre_hooks) == hook_count
        assert len(scores) == meta['visual_tokens'] and not text._qprobe_active
        assert stats['completed_decoder_layers'] == 3 and not stats['probe_cache_created']
        assert bank.fingerprint() == digest
        checks.append(dict(name='early_exit_no_later_layers_no_persistent_cache', layer_entries=layers, stats=stats))
        full_nodes, _ = select_half(nodes, scores, full=True)
        e2, p2, i2, m2 = context(model, builder, full_nodes, question,
            limits=Limits(2048, 384, 16, 128), max_new_tokens=8)
        assert torch.equal(emb, e2) and torch.equal(pos, p2) and torch.equal(ids, i2) and meta == m2
        checks.append(dict(name='100_percent_readout_exact_layout', exact=True))
        try:
            with patch('probe.attention_rows', side_effect=RuntimeError('intentional contract fault')):
                score_visual(model, builder, question, emb, pos, ids)
        except RuntimeError as error:
            assert str(error) == 'intentional contract fault'
        else:
            raise AssertionError('probe swallowed an unrelated failure')
        assert len(attn._forward_pre_hooks) == hook_count and not text._qprobe_active
        checks.append(dict(name='failure_removes_hook_propagates_error', exact=True))
        baseline = text(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state
        score_visual(model, builder, question, emb, pos, ids)
        repeated = text(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state
        assert torch.equal(baseline, repeated)
        checks.append(dict(name='normal_forward_after_probe', exact=True))
        bank.clear();old.clear()

        # ALL303 layouts. Random scores are only a budget/order fixture.
        from score_memory_eval import prompt
        groups = read(OUT/'inputs/groups.json');questions = read(OUT/'inputs/questions.json');layouts = []
        for group in groups:
            bank = Banks()
            for q in sorted([q for q in questions if q['group'] == group['id']], key=lambda q: (q['pairs'], q['id'])):
                while bank.pairs < q['pairs']:
                    i = bank.pairs
                    a, z = group['frames'][2*i], group['frames'][min(2*i+1, len(group['frames'])-1)]
                    torch.manual_seed(400+i//5)
                    bank.append(torch.randn(48, 32), i, a['pts_s'], z['pts_s'])
                before = bank.fingerprint();full_nodes = bank.full()
                for arm in ARMS:
                    nodes = bank.export(arm) if arm != 'LS_probe24' else select_half(
                        full_nodes, torch.randn(sum(len(n.features) for n in full_nodes)))[0]
                    _, positions, token_ids, metadata = context(model, builder, nodes, prompt(q),
                        limits=Limits(2048, 384, 16, 128), max_new_tokens=8)
                    question_rows(builder, prompt(q), token_ids.shape[1])
                    assert metadata['visual_tokens'] <= (384 if arm == 'LS_full48' else 192)
                    assert torch.equal(positions[0, 0], torch.arange(token_ids.shape[1]))
                    layouts.append(dict(id=q['id'], arm=arm, input_tokens=metadata['input_tokens'],
                                        visual_tokens=metadata['visual_tokens'], question_text_rows_checked=True))
                assert before == bank.fingerprint()
            bank.clear()
        assert len(layouts) == 909
        checks.append(dict(name='all_frozen303_question_layouts', layouts=909))
    save(OUT/'CPU_preflight.json', dict(status='passed', checks=checks, layout_rows=layouts,
        seconds=time.monotonic()-started, pretrained_loads=0, GPU_jobs=0,
        random_weights_are_not_capability_evidence=True,
        groups_sha256=sha(OUT/'inputs/groups.json'), questions_sha256=sha(OUT/'inputs/questions.json')))
    print(dict(status='passed', checks=len(checks), layouts=909, seconds=time.monotonic()-started), flush=True)


if __name__ == '__main__':main()
