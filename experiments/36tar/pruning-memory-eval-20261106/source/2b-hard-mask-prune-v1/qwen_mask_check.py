"""Qwen tiny-weight integration; no pretrained ability claim."""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-learned-prune-main-v1'))
from protocol import runtime, make_builder, read, PREP
from hard_prune import torch, GroupScorer, prefix, tail, selected_ids, policy_attention
from objective import last_logits, option_ids, loss_parts


def check_visual(visual, pixels, grid, scorer, previous=None, atol=3e-6, rtol=3e-5):
    observed, x, positions = prefix(visual, pixels, grid)
    scores, state = scorer(observed, previous)
    ids = selected_ids(scores, 54)
    stats = {}
    with torch.no_grad():
        gathered = tail(visual, x, positions, scores, ids, 'gather')
        masked = tail(visual, x, positions, scores, ids, 'mask')
        torch.testing.assert_close(masked, gathered, atol=atol, rtol=rtol)
        stats['mask_gather_max_abs'] = float((masked - gathered).abs().max())
        stats['mask_gather_relative_l2'] = float((masked - gathered).norm() / gathered.norm().clamp_min(1e-12))
        changed = tail(visual, x, positions, scores * 2 + 7, ids, 'mask')
        assert torch.equal(changed, masked), 'continuous scores affect forward despite unchanged IDs'
        keep = torch.zeros(108, dtype=torch.bool, device=x.device).scatter(0, ids, True).repeat_interleave(4)
        altered = x.clone()
        altered[~keep] += 100 * torch.randn_like(altered[~keep])
        isolated = tail(visual, altered, positions, scores, ids, 'mask')
        torch.testing.assert_close(isolated, masked, atol=atol, rtol=rtol)
        stats['isolation_max_abs'] = float((isolated - masked).abs().max())
        all_ids = torch.arange(108, device=x.device)
        all_masked = tail(visual, x, positions, scores, all_ids, 'mask')
        all_gathered = tail(visual, x, positions, scores, all_ids, 'gather')
        torch.testing.assert_close(all_masked, all_gathered, atol=atol, rtol=rtol)
        stats['all_keep_max_abs'] = float((all_masked - all_gathered).abs().max())
    return stats, state


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    p = argparse.ArgumentParser();p.add_argument('--out', type=Path, required=True);a = p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20261102)
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B', local_files_only=True)
    cfg = env['tiny_config'](processor);cfg['vision_config']['depth'] = 24
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg, attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder = make_builder(env, processor);scorer = GroupScorer(model.config.vision_config.hidden_size)
    grid = torch.tensor([[1, 18, 24]]);state = None;checks = [];records = []
    with env['fp32_reference']():
        for i in range(2):
            pixels = torch.randn(432, 1536)
            check, state = check_visual(model.model.visual, pixels, grid, scorer, state)
            checks.append(check)
            observed, x, pos = prefix(model.model.visual, pixels, grid)
            scores, _ = scorer(observed, None if i == 0 else previous)
            scores.retain_grad();ids = selected_ids(scores, 54)
            feat = tail(model.model.visual, x, pos, scores, ids, 'mask')
            records.append((i, float(i), i+.5, feat, ids));previous = state
        q = read(PREP/'train.json')[0]
        logits, tokens = last_logits(model, builder, records, q)
        options = option_ids(builder, logits.device)
        # Force a correct teacher only to exercise KL gradients, not assess quality.
        teacher = torch.zeros(1, 5);teacher[0, q['answer_index']] = 2
        loss, ce, kl, enabled = loss_parts(logits, options, q['answer_index'], teacher)
        assert enabled and torch.isfinite(loss)
        loss.backward()
        grad = float(torch.sqrt(sum(p.grad.square().sum() for p in scorer.parameters() if p.grad is not None)))
        assert grad > 0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in scorer.parameters())
        assert all(p.grad is None for p in model.parameters())
        assert scores.grad.abs().sum() > 0
        wrong = teacher.roll(1, -1)
        _, _, wrong_kl, enabled_wrong = loss_parts(logits.detach(), options, q['answer_index'], wrong)
        assert not enabled_wrong and float(wrong_kl) == 0
        extreme_scores = torch.tensor([-10000., 10000.], requires_grad=True)
        extreme_logits = torch.randn(2, 8, 8) * 10000
        weights = policy_attention(extreme_logits, extreme_scores, torch.tensor([0]), True)
        assert torch.isfinite(weights).all() and torch.equal(weights[..., 4:], torch.zeros_like(weights[..., 4:]))
        weights.square().sum().backward()
        assert torch.isfinite(extreme_scores.grad).all()
        assert torch.equal(selected_ids(torch.zeros(108), 54), torch.arange(54))
    result = dict(status='passed_tiny_random_weights_only', visual_checks=checks,
                  tokens=tokens, ce=float(ce.detach()), kl=float(kl.detach()), scorer_gradient_norm=grad,
                  base_frozen=True, extreme_mask_checked=True, pretrained_loads=0, GPU=0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
