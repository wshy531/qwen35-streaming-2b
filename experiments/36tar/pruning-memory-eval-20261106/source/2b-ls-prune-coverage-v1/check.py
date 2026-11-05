"""Check causal schedule replay and bank bounds on CPU, without model weights."""
import json
import os

from matched_ls import Bank, MatchedLS, sparse_pair, structure
from resolution_nodes import digest
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(20261103)
matched = MatchedLS(('fixed', 'content', 'ce'))
baseline = Bank('LS', 108, budget=864, recent=432, closed=324)
independent = Bank('LS', 54, budget=432, recent=216, closed=162)
grid = torch.arange(108)
merges = diverged = 0
for pair in range(64):
    features = torch.randn(108, 16, generator=generator)
    full = sparse_pair(features, grid, pair, float(2 * pair), float(2 * pair + 1))
    indices = (grid[::2], grid[1::2], torch.randperm(108, generator=generator)[:54].sort().values)
    nodes = {name: sparse_pair(features[ids], ids, pair, full.begin, full.end)
             for name, ids in zip(matched.pruned, indices)}
    starts = pair % 7 == 0
    baseline.append(full, starts_event=starts)
    independent.append(nodes['fixed'], starts_event=starts)
    merges += len(matched.append(full, nodes, starts_event=starts))
    diverged += structure(independent) != structure(matched.full)
    assert digest(baseline.nodes()) == digest(matched.full.nodes())
    assert all(structure(bank) == structure(baseline) for bank in matched.pruned.values())
    assert all(bank.status()['tokens'] <= 432 and all(n.last <= pair for n in bank.nodes())
               for bank in matched.pruned.values())
    assert all(bank.status()['recent_tokens'] == 54 * min(4, pair + 1) for bank in matched.pruned.values())
assert merges == 56
print(json.dumps(dict(passed=True, steps=64, matched_view_steps=192, reference_merges=merges,
    independent_same_capacity_structure_differed_steps=diverged, full_bank_parity_steps=64, GPU=0)))
