"""Recompute one visual pair at a time after CPU-leaf/QA backpropagation.

Preserves the existing fixed LS schedule and hard-mask STE, not a derivative
through discrete selection. CPU cache grows with the bounded training clip.
No optimizer step is permitted between capture and the final visual backward.
"""
from dataclasses import dataclass
import sys
import torch
from replay import ROOT, Replay, banks
from delayed_qa import training_context, last_logits

for name in ('2b-hard-mask-prune-v1', '2b-midprune-memory-v1',
             '2b-learned-prune-main-v1'):
    sys.path.insert(0, str(ROOT / 'experiments' / name))
from hard_prune import encode
from adapter import sparse_pair
from objective import loss_parts, option_ids


@dataclass
class CachedPair:
    previous: object
    ids: torch.Tensor
    features: torch.Tensor


@dataclass
class Capture:
    pairs: dict
    nodes: list
    replay: Replay
    versions: dict


def assert_unchanged(scorer, captured):
    if {n: p._version for n, p in scorer.named_parameters()} != captured.versions:
        raise RuntimeError('scorer changed between capture and recomputation')


def capture_pairs(visual, scorer, pairs):
    """pairs yields (begin, end, pixel_values, grid); pixels can stay on CPU."""
    if visual.training or any(p.requires_grad for p in visual.parameters()):
        raise ValueError('visual base must be frozen and in eval mode')
    device = next(scorer.parameters()).device
    captured = Capture({}, [], Replay(), {n: p._version for n, p in scorer.named_parameters()})
    bank = banks.Bank('LS', 54, budget=864, recent=432, closed=324)
    previous = None
    with torch.no_grad(), captured.replay.capture():
        for i, (begin, end, pixels, grid) in enumerate(pairs):
            saved_previous = None if previous is None else previous.detach().cpu().clone()
            features, ids, _, previous = encode(
                visual, pixels.to(device), grid.to(device), scorer,
                previous=previous, mode='mask')
            captured.pairs[i] = CachedPair(saved_previous, ids.cpu().clone(),
                                           features.detach().cpu().clone())
            bank.append(sparse_pair(features, ids, i, begin, end), starts_event=i % 3 == 0)
    captured.nodes = bank.export()
    assert_unchanged(scorer, captured)
    return captured


def qa_leaf_gradients(model, builder, captured, question, answer_index):
    """Only CPU projected leaves plus language graph; releases graph on return."""
    leaves = {i: p.features.detach().clone().requires_grad_(True)
              for i, p in captured.pairs.items()}
    emb, pos, ids, meta = training_context(model, builder, captured.nodes, question,
                                          captured.replay, leaves)
    logits = last_logits(model, emb, pos)
    loss, _, _, _ = loss_parts(logits, option_ids(builder, logits.device), answer_index)
    loss.backward()
    gradients = {i: leaf.grad.detach().cpu().clone() for i, leaf in leaves.items()
                 if leaf.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    return dict(loss=float(loss.detach()), logits=logits.detach().cpu(),
                ids=ids.detach().cpu(), positions=pos.detach().cpu(), metadata=meta,
                gradients=gradients)


def visual_backward(visual, scorer, captured, gradients, pair_at):
    """pair_at(i) -> (begin, end, CPU pixels, grid), deterministic across passes."""
    assert_unchanged(scorer, captured)
    device = next(scorer.parameters()).device
    score_gradients = {}
    for i, gradient in sorted(gradients.items()):
        assert_unchanged(scorer, captured)
        saved = captured.pairs[i]
        _, _, pixels, grid = pair_at(i)
        previous = None if saved.previous is None else saved.previous.to(device)
        features, ids, scores, state = encode(
            visual, pixels.to(device), grid.to(device), scorer,
            previous=previous, mode='mask')
        assert torch.equal(ids.cpu(), saved.ids), 'recomputed selection changed'
        torch.testing.assert_close(features.detach().cpu(), saved.features, atol=2e-5, rtol=5e-5)
        scores.retain_grad()
        features.backward(gradient.to(features))
        assert scores.grad is not None and torch.isfinite(scores.grad).all()
        score_gradients[i] = scores.grad.detach().cpu().clone()
        # Do not retain any per-pair autograd graph into the next iteration.
        del features, ids, scores, state, previous
    assert_unchanged(scorer, captured)
    assert all(p.grad is None for p in visual.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scorer.parameters())
    return score_gradients


def cache_bytes(captured):
    return sum(t.numel() * t.element_size() for p in captured.pairs.values()
               for t in (p.previous, p.ids, p.features) if t is not None)
