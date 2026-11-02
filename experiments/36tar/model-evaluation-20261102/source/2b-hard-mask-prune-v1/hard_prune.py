"""Fixed-budget, causal visual pruning. Frozen Qwen; experimental STE only."""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-learned-prune-mechanism-v1'))
from bias_prune import GroupScorer


def selected_ids(scores, keep):
    if scores.ndim != 1 or not 1 <= keep <= len(scores):
        raise ValueError('invalid fixed group budget')
    if not torch.isfinite(scores).all():
        raise ValueError('non-finite selection scores')
    return torch.argsort(scores.detach(), descending=True, stable=True)[:keep].sort().values


def policy_attention(logits, scores, ids, surrogate):
    """Exact hard forward, bounded softmax surrogate on scores only.

    Q/K receive the ordinary hard-mask gradient. The auxiliary relaxation has
    detached Q/K, preventing an extra soft Q/K gradient. This is a biased STE,
    not the derivative of argsort. No epsilon resurrects a deleted key.
    """
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter(0, ids, True)
    keep = keep.repeat_interleave(4)
    hard = logits.masked_fill(~keep, float('-inf')).softmax(-1)
    if surrogate and torch.is_grad_enabled():
        relaxed = (logits.detach() + F.logsigmoid(scores).repeat_interleave(4)).softmax(-1)
        hard = hard + (relaxed - relaxed.detach())
    return hard


def masked_block(block, x, positions, scores, ids, surrogate=True):
    attn = block.attn
    n = len(x)
    assert n == 4 * len(scores)
    rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb_vision
    q, k, v = attn.qkv(block.norm1(x)).reshape(n, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = rope(q, k, *positions)
    q, k, v = [a.transpose(0, 1) for a in (q, k, v)]
    logits = (q @ k.transpose(-1, -2)) * attn.scaling
    weights = policy_attention(logits, scores, ids, surrogate)
    y = weights @ v
    x = x + attn.proj(y.transpose(0, 1).reshape(n, -1))
    return x + block.mlp(block.norm2(x))


def prefix(visual, pixels, grid):
    if visual.training or grid.tolist() != [[1, 18, 24]] or len(visual.blocks) != 24:
        raise ValueError('requires frozen 24-layer visual and one 288x384 pair')
    with torch.no_grad():
        x = visual.patch_embed(pixels) + visual.fast_pos_embed_interpolate(grid)
        r = visual.rot_pos_emb(grid).reshape(len(x), -1)
        emb = torch.cat((r, r), -1)
        positions = (emb.cos(), emb.sin())
        cu = torch.tensor([0, len(x)], device=x.device, dtype=torch.int32)
        for block in visual.blocks[:11]:
            x = block(x, cu_seqlens=cu, position_embeddings=positions)
        observed = x
        x = visual.blocks[11](x, cu_seqlens=cu, position_embeddings=positions)
    return observed, x, positions


def tail(visual, x, positions, scores, ids, mode):
    patches = (ids[:, None] * 4 + torch.arange(4, device=x.device)).flatten()
    if mode == 'gather':
        x = x[patches]
        positions = tuple(p[patches] for p in positions)
        cu = torch.tensor([0, len(x)], device=x.device, dtype=torch.int32)
        for block in visual.blocks[12:]:
            x = block(x, cu_seqlens=cu, position_embeddings=positions)
    elif mode in ('mask', 'mask_no_surrogate'):
        for block in visual.blocks[12:]:
            x = masked_block(block, x, positions, scores, ids, mode == 'mask')
        x = x[patches]
    else:
        raise ValueError(mode)
    return visual.merger(x)


def encode(visual, pixels, grid, scorer, *, previous=None, keep=54, mode='gather'):
    observed, x, pos = prefix(visual, pixels, grid)
    scores, state = scorer(observed, previous)
    ids = selected_ids(scores, keep)
    return tail(visual, x, pos, scores, ids, mode), ids, scores, state
