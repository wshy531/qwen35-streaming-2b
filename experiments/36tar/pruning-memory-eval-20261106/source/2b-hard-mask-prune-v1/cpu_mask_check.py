"""Algebra/gradient feasibility only. No Qwen weights, no ability or speed claim."""
import json
from pathlib import Path
import torch
from torch import nn


def straight_through_mask(scores, keep):
    ids = torch.argsort(scores, descending=True, stable=True)[:keep].sort().values
    hard = torch.zeros_like(scores).scatter(0, ids, 1)
    soft = scores.sigmoid()
    # Forward is exactly hard; backward is a biased sigmoid surrogate.
    return hard + (soft - soft.detach()), ids


class Block(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x, positions, policy=None):
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        # Small coordinate-dependent rotation, retained coordinates never renumbered.
        angle = positions[:, None] * torch.arange(1, 9, dtype=x.dtype)[None, :] / 32
        def rotate(z):
            a, b = z.chunk(2, dim=-1)
            return torch.cat((a * angle.cos() - b * angle.sin(),
                              a * angle.sin() + b * angle.cos()), dim=-1)
        weights = (rotate(q) @ rotate(k).T / 4).softmax(-1)
        if policy is not None:
            # No additive epsilon leakage. At least one group must remain.
            weights = weights * policy[None, :]
            weights = weights / weights.sum(-1, keepdim=True)
        x = x + self.proj(weights @ v)
        return x + self.ff(x)


def run(dtype, groups, keep):
    torch.manual_seed(20261102)
    blocks = nn.ModuleList([Block() for _ in range(3)]).to(dtype).eval()
    blocks.requires_grad_(False)
    x = torch.randn(groups * 4, 16, dtype=dtype)
    positions = torch.arange(len(x), dtype=dtype)
    scores = torch.randn(groups, dtype=dtype, requires_grad=True)
    mask, ids = straight_through_mask(scores, keep)
    patches = (ids[:, None] * 4 + torch.arange(4)).flatten()
    policy = mask.repeat_interleave(4)

    def dense(inputs, gate):
        z = inputs
        for block in blocks:
            z = block(z, positions, gate)
        return z[patches].reshape(keep, 4, 16).mean(1)

    masked = dense(x, policy)
    gathered = x[patches]
    for block in blocks:
        gathered = block(gathered, positions[patches])
    gathered = gathered.reshape(keep, 4, 16).mean(1)
    error = (masked - gathered).abs().max().item()
    changed = x.clone()
    changed[policy.detach() == 0] += 3 * torch.randn_like(changed[policy.detach() == 0])
    isolation_error = (dense(changed, policy) - masked).abs().max().item()
    masked.square().mean().backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0
    limit = 3e-6 if dtype == torch.float32 else 1e-12
    assert error < limit and isolation_error < limit
    assert mask.detach().sum().item() == keep
    _, ties = straight_through_mask(torch.zeros(groups, dtype=dtype), keep)
    assert torch.equal(ties, torch.arange(keep))
    return dict(dtype=str(dtype), groups=groups, keep=keep,
                mask_gather_max_abs=error, dropped_value_isolation_max_abs=isolation_error,
                scorer_gradient_norm=scores.grad.norm().item(),
                dropped_score_gradient_norm=scores.grad[mask.detach() == 0].norm().item())


if __name__ == '__main__':
    torch.set_num_threads(2)
    rows = [run(dtype, groups, keep) for dtype in (torch.float32, torch.float64)
            for groups, keep in ((8, 4), (8, 8), (108, 54))]
    report = dict(scope='synthetic CPU algebra only; not Qwen integration or true top-k derivative',
                  checks=rows)
    target = Path(__file__).with_name('cpu_mask_check.json')
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
