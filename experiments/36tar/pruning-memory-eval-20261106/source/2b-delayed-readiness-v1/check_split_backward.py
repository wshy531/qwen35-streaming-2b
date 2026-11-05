"""Compare whole-graph and recomputed gradients using actual tiny Qwen modules."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from split_backward import (capture_pairs, qa_leaf_gradients, visual_backward,
                            cache_bytes, assert_unchanged)
from check_delayed_qa import (ROOT, Replay, banks, training_context, last_logits,
                              sparse_pair, GroupScorer, loss_parts, option_ids,
                              runtime, make_builder)
from hard_prune import encode
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    started = time.monotonic()
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(ROOT / 'models/Qwen3.5-2B', local_files_only=True)
    builder = make_builder(env, processor)
    config = env['tiny_config'](processor)
    config['vision_config']['depth'] = 24
    model = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**config, attn_implementation='sdpa')).float().eval().requires_grad_(False)
    scorer = GroupScorer(model.config.vision_config.hidden_size)
    versions = {n: p._version for n, p in model.named_parameters()}
    pairs = [(2*i, 2*i+1, torch.randn(432, 1536), torch.tensor([[1, 18, 24]])) for i in range(24)]
    question = 'What was at the beginning? A. Cup B. Book C. Chair D. Lamp E. Bowl'
    with env['fp32_reference']():
        replay = Replay()
        bank = banks.Bank('LS', 54, budget=864, recent=432, closed=324)
        leaves, scores = {}, {}
        state = None
        with replay.capture():
            for i, (begin, end, pixels, grid) in enumerate(pairs):
                features, kept, scores[i], state = encode(model.model.visual, pixels, grid,
                                                          scorer, previous=state, mode='mask')
                scores[i].retain_grad()
                features.retain_grad()
                leaves[i] = features
                bank.append(sparse_pair(features, kept, i, begin, end), starts_event=i % 3 == 0)
        nodes = bank.export()
        emb, pos, ids, meta = training_context(model, builder, nodes, question, replay, leaves)
        logits = last_logits(model, emb, pos)
        loss, _, _, _ = loss_parts(logits, option_ids(builder, logits.device), 0)
        loss.backward()
        expected = {n: p.grad.clone() for n, p in scorer.named_parameters()}
        expected_scores = {i: s.grad.clone() for i, s in scores.items() if s.grad is not None}
        expected_leaves = {i: f.grad.clone() for i, f in leaves.items() if f.grad is not None}
        expected_logits, expected_loss = logits.detach().clone(), float(loss.detach())
        del loss, logits, emb, leaves, scores, features, state
        scorer.zero_grad(set_to_none=True)
        captured = capture_pairs(model.model.visual, scorer, pairs)
        assert captured.replay.steps == replay.steps
        assert [(n.first,n.last) for n in captured.nodes] == [(n.first,n.last) for n in nodes]
        observed = qa_leaf_gradients(model, builder, captured, question, 0)
        assert torch.equal(ids, observed['ids']) and torch.equal(pos, observed['positions'])
        assert meta == observed['metadata'] and abs(expected_loss-observed['loss']) <= 1e-6
        torch.testing.assert_close(expected_logits, observed['logits'], atol=1e-6, rtol=1e-6)
        assert expected_leaves.keys() == observed['gradients'].keys()
        for i, gradient in observed['gradients'].items():
            torch.testing.assert_close(gradient, expected_leaves[i], atol=1e-7, rtol=1e-5)
        obtained_scores = visual_backward(model.model.visual, scorer, captured,
                                           observed['gradients'], pairs.__getitem__)
        deltas = {}
        for n, p in scorer.named_parameters():
            torch.testing.assert_close(p.grad, expected[n], atol=1e-7, rtol=1e-4)
            deltas[n] = float((p.grad-expected[n]).abs().max())
        for i, gradient in obtained_scores.items():
            torch.testing.assert_close(gradient, expected_scores[i], atol=1e-7, rtol=1e-4)
        assert obtained_scores[0].abs().sum() > 0
        assert all(p.grad is None and p._version == versions[n] for n,p in model.named_parameters())
        # Reject even a no-op optimizer-style in-place write between passes.
        with torch.no_grad():
            next(scorer.parameters()).add_(0)
        try:
            assert_unchanged(scorer, captured)
        except RuntimeError:
            mutation_guard = True
        else:
            raise AssertionError('missing mutation guard')
    result = dict(status='passed_tiny_random_weights', pairs=24, merges=len(replay.steps),
                  gradient_atol=1e-7, gradient_rtol=1e-4, param_max_abs_delta=deltas,
                  first_score_grad_l1=float(obtained_scores[0].abs().sum()),
                  ce=observed['loss'], cache_tensor_bytes=cache_bytes(captured),
                  mutation_guard=mutation_guard, frozen_base=True, optimizer_steps=0,
                  GPU=0, seconds=time.monotonic()-started,
                  limits='CPU synthetic pixels, fixed schedule and STE; GPU memory savings unmeasured',
                  sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), Path(__file__).with_name('split_backward.py')]})
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
