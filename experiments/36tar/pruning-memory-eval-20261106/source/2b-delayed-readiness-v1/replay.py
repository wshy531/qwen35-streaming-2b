"""Differentiable value replay of a fixed, existing LS merge schedule.

This is a training preparation helper. Inference Bank remains the authority for
event boundaries, merge destinations, masses, coordinates and selected nodes.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-resolution-cost-v1'))
import torch
import resolution_banks_base as banks
from resolution_nodes import metric


def key(node):
    return node.first, node.last


@dataclass
class MergeStep:
    left: tuple
    right: tuple
    chosen: list
    unmerged: list
    destination: list
    mass: list
    permutation: list


class Replay:
    def __init__(self):
        self.steps = {}

    @contextmanager
    def capture(self):
        original = banks.merge

        def traced(left, right, within=False):
            result = original(left, right, within)
            features = torch.cat((left.features, right.features))
            mass = torch.cat((left.mass, right.mass))
            coords = torch.cat((left.coords, right.coords))
            values = metric(features)
            best, destination = (values[::2] @ values[1::2].T).max(1)
            order = torch.argsort(best, descending=True, stable=True)
            chosen, unmerged = order[:len(features)//2], order[len(features)//2:]
            output_coords = torch.cat((coords[::2][unmerged], coords[1::2]))
            h, w = left.grid
            permutation = torch.argsort(output_coords[:, 0]*(h*w) + output_coords[:, 1]*w
                                        + output_coords[:, 2], stable=True)
            assert key(result) not in self.steps
            self.steps[key(result)] = MergeStep(key(left), key(right), chosen.tolist(),
                                               unmerged.tolist(), destination.tolist(),
                                               mass.tolist(), permutation.tolist())
            return result

        banks.merge = traced
        try:
            yield self
        finally:
            banks.merge = original

    def values(self, nodes, leaves):
        cache = {}

        def compute(span):
            if span in cache:
                return cache[span]
            if span[0] == span[1]:
                result = leaves[span[0]]
            else:
                step = self.steps[span]
                x = torch.cat((compute(step.left), compute(step.right)))
                mass = torch.tensor(step.mass, dtype=x.dtype, device=x.device)
                chosen = torch.tensor(step.chosen, dtype=torch.long, device=x.device)
                unmerged = torch.tensor(step.unmerged, dtype=torch.long, device=x.device)
                destination = torch.tensor(step.destination, dtype=torch.long, device=x.device)[chosen]
                weighted = (x[1::2] * mass[1::2]).scatter_add(
                    0, destination[:, None].expand(-1, x.shape[1]), x[::2][chosen] * mass[::2][chosen])
                weights = mass[1::2].scatter_add(0, destination[:, None], mass[::2][chosen])
                permutation = torch.tensor(step.permutation, dtype=torch.long, device=x.device)
                result = torch.cat((x[::2][unmerged], weighted / weights))[permutation]
            cache[span] = result
            return result

        return [compute(key(node)) for node in nodes]

