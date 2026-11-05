"""CPU-only factorial capacity contract; no model quality claims."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-midprune-memory-v1'))
from adapter import sparse_pair, Bank
from resolution_nodes import digest
import torch


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(20261103)
    # R4H4 and R8H8 reproduce prior auto432/auto864. Add the crossed cells.
    banks = {(r, h): Bank('LS', 54, budget=54*(r+h), recent=54*r, closed=162)
             for r in (4, 8) for h in (4, 8)}
    old = {4: Bank('LS', 54, budget=432, recent=216, closed=162),
           8: Bank('LS', 54, budget=864, recent=432, closed=162)}
    checks = 0
    ids = torch.arange(0, 108, 2)
    for i in range(96):
        node = sparse_pair(torch.randn(54, 16, generator=gen), ids, i, 2*i, 2*i+1)
        start = i % 7 == 0
        for bank in [*banks.values(), *old.values()]:
            bank.append(node, starts_event=start)
            assert bank.status()['tokens'] <= bank.budget
            assert all(n.last <= i for n in bank.nodes())
        for r in (4, 8):
            a, b = banks[r, 4], banks[r, 8]
            assert digest([e.node for e in a.recent]) == digest([e.node for e in b.recent])
            assert len(a.recent) == min(r, i+1)
            assert digest(banks[r, r].nodes()) == digest(old[r].nodes())
            checks += 1
        for (r, h), bank in banks.items():
            assert len(bank.history) <= h
            before = digest(bank.nodes())
            exported = bank.export()
            exported[-1].features.zero_()
            assert digest(bank.nodes()) == before
    report = dict(passed=True, CPU_only=True, synthetic_steps=96,
        same_recent_comparisons=checks, legacy_endpoint_parity=checks,
        configurations={f'R{r}H{h}': dict(tokens=b.budget, recent_pairs=r, history_nodes=h)
                        for (r, h), b in banks.items()},
        interpretation='History capacity intervention includes changed merge schedule and token length; not pure semantic interference.',
        real_weight_answers=0)
    out = ROOT / 'runs/ls_prune_expansion_prepare_20261103_01/CPU_capacity.json'
    with out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
