"""Small CPU test of actual LS merge values and an old-leaf score gradient."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for name in ('2b-hard-mask-prune-v1', '2b-midprune-memory-v1'):
    sys.path.insert(0, str(ROOT / 'experiments' / name))
from replay import Replay, banks
from adapter import sparse_pair
from hard_prune import GroupScorer, policy_attention, selected_ids
import torch


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    scorer = GroupScorer(16)
    previous = None
    leaves, scores_by_pair = {}, {}
    bank = banks.Bank('LS', 54, budget=864, recent=432, closed=324)
    replay = Replay()
    checks = []
    original_merge = banks.merge
    with replay.capture():
        for index in range(24):
            # Synthetic frozen visual activations exercise the real STE helper.
            x = torch.randn(432, 16)
            scores, previous = scorer(x, previous)
            scores.retain_grad()
            ids = selected_ids(scores, 54)
            weights = policy_attention((x @ x.T) / 4, scores, ids, True)
            projected = (weights @ x).reshape(108, 4, 16).mean(1)[ids]
            leaves[index], scores_by_pair[index] = projected, scores
            bank.append(sparse_pair(projected, ids, index, 2*index, 2*index+1),
                        starts_event=index % 3 == 0)
            nodes = bank.export()
            actual = replay.values(nodes, leaves)
            delta = max(float((a.detach()-n.features).abs().max()) for a, n in zip(actual, nodes))
            assert all(torch.allclose(a.detach(), n.features, atol=1e-6, rtol=1e-6)
                       for a, n in zip(actual, nodes))
            checks.append(dict(pair=index, merges=len(replay.steps), max_abs_difference=delta,
                               tokens=sum(len(x) for x in actual)))
    assert banks.merge is original_merge
    history = [entry.node for entry in bank.history]
    assert history and history[0].first == 0 and history[0].last > 0
    replayed = replay.values(history, leaves)
    loss = sum(x.square().mean() for x in replayed)
    loss.backward()
    old_gradient = float(scores_by_pair[0].grad.abs().sum())
    assert old_gradient > 0 and torch.isfinite(scores_by_pair[0].grad).all()
    assert all(scores_by_pair[i].grad is None for i in range(16, 24))
    assert any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in scorer.parameters())
    report = dict(passed=True, CPU_only=True, synthetic_pairs=24, checks=checks,
                  scorer_width=16, old_pair0_score_gradient_l1=old_gradient,
                  fixed_schedule_merges=len(replay.steps), recent_pairs_have_no_history_only_gradient=True,
                  old_inference_history_requires_grad=any(n.features.requires_grad for n in history),
                  loss='synthetic history squared mean, not language QA',
                  real_weight_forward=0, GPU=0,
                  remaining=['real visual and language QA parity', 'gradient across QA context insertion',
                             'bounded recomputation memory', 'timestamp-supervised training data'],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path(__file__).with_name('replay.py')]})
    output = ROOT / 'runs/delayed_readiness_prepare_20261104_01/CPU_replay.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','source_sha256')}))


if __name__ == '__main__':
    main()
