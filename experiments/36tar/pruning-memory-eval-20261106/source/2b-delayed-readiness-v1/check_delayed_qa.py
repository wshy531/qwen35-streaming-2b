"""CPU integration check with random tiny Qwen weights and synthetic pixels."""
import json
import os
import sys
from pathlib import Path

from replay import ROOT, Replay, banks
from delayed_qa import last_logits, training_context
from resolution_runtime import context, make_builder

for name in ('2b-hard-mask-prune-v1', '2b-midprune-memory-v1',
             '2b-learned-prune-main-v1'):
    sys.path.insert(0, str(ROOT / 'experiments' / name))
from adapter import sparse_pair
from hard_prune import GroupScorer, prefix, selected_ids, tail
from objective import loss_parts, option_ids
from protocol import runtime
import torch


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(
        ROOT / 'models/Qwen3.5-2B', local_files_only=True)
    config = env['tiny_config'](processor)
    config['vision_config']['depth'] = 24
    model = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**config, attn_implementation='sdpa'))
    model.float().eval().requires_grad_(False)
    builder = make_builder(env, processor)
    scorer = GroupScorer(model.config.vision_config.hidden_size)
    grid = torch.tensor([[1, 18, 24]])
    replay = Replay()
    bank = banks.Bank('LS', 54, budget=864, recent=432, closed=324)
    leaves, scores = {}, {}
    state = None
    with env['fp32_reference']():
        with replay.capture():
            for i in range(24):
                observed, activation, pos = prefix(model.model.visual, torch.randn(432, 1536), grid)
                scores[i], state = scorer(observed, state)
                scores[i].retain_grad()
                kept = selected_ids(scores[i], 54)
                leaves[i] = tail(model.model.visual, activation, pos, scores[i], kept, 'mask')
                bank.append(sparse_pair(leaves[i], kept, i, 2*i, 2*i+1),
                            starts_event=i % 3 == 0)
        nodes = bank.export()
        assert nodes[0].first == 0 and nodes[0].last > 0
        assert nodes[-1].last == 23 and replay.steps
        question = 'What was at the beginning? A. Cup B. Book C. Chair D. Lamp E. Bowl'
        emb, pos, ids, metadata = training_context(model, builder, nodes, question, replay, leaves)
        with torch.no_grad():
            ref_emb, ref_pos, ref_ids, ref_metadata = context(model, builder, nodes, question)
            reference = last_logits(model, ref_emb, ref_pos)
        assert torch.equal(ids, ref_ids) and torch.equal(pos, ref_pos) and metadata == ref_metadata
        torch.testing.assert_close(emb, ref_emb, atol=1e-6, rtol=1e-6)
        logits = last_logits(model, emb, pos)
        torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)
        loss, _, _, _ = loss_parts(logits, option_ids(builder, logits.device), 0)
        loss.backward()
        first_gradient = float(scores[0].grad.abs().sum())
        assert first_gradient > 0 and torch.isfinite(scores[0].grad).all()
        assert all(p.grad is None for p in model.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in scorer.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scorer.parameters())
    result = dict(status='passed_tiny_random_weights', pairs=24, merges=len(replay.steps),
                  first_node_span=[nodes[0].first, nodes[0].last], tokens=ids.numel(),
                  visual_tokens=metadata['visual_tokens'], first_score_grad_l1=first_gradient,
                  max_embedding_delta=float((emb.detach()-ref_emb).abs().max()),
                  max_logit_delta=float((logits.detach()-reference).abs().max()),
                  ce=float(loss.detach()), frozen_base=True, pretrained_loads=0, GPU=0)
    out = ROOT / 'runs/delayed_readiness_prepare_20261104_01/CPU_delayed_QA.json'
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
