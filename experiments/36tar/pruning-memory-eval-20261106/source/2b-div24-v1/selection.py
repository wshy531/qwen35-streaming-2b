"""Pair-local Qwen projection adaptation of DivPrune Algorithm 1.

Reference: https://arxiv.org/html/2503.02175v2 (Algorithm 1).
No author environment/code is imported. Greedy heuristic, not an optimality claim.
"""
import torch
import torch.nn.functional as F


@torch.inference_mode()
def div_indices(features, keep=24):
    if features.ndim != 2 or features.shape[0] != 48 or not 1 <= keep <= 48:
        raise ValueError('complete 48-token pair and bounded subset required')
    if features.device.type != 'cpu' or not torch.isfinite(features).all():
        raise ValueError('finite CPU projection required')
    metric = F.normalize(features.float(), dim=-1, eps=1e-12)
    distance = (1 - metric @ metric.T).clamp(0, 2)
    distance.fill_diagonal_(float('inf'))
    # Algorithm 1 first stage: largest nearest-neighbour distance, excluding self.
    seed = int(distance.min(dim=1).values.argmax())
    chosen = [seed]
    nearest = distance[:, seed].clone()
    nearest[seed] = -float('inf')
    while len(chosen) < keep:
        index = int(nearest.argmax())  # lowest original index wins ties
        chosen.append(index)
        nearest = torch.minimum(nearest, distance[:, index])
        nearest[chosen] = -float('inf')
    return torch.tensor(sorted(chosen), dtype=torch.long)


def grid_indices():
    return torch.tensor([i for i in range(48) if (i // 8 + i % 8) % 2 == 0])
