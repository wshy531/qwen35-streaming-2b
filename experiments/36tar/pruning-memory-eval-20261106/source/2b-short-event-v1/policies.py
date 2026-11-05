"""B02: cap within-event folding; keep the existing closed-root merger unchanged."""
from collections import deque
import random

from memory384 import Forest384

CAP_PAIRS = 4
RESERVOIR_SEED = 20261020


def block(node):
    from legacy_sources import load
    return load().features.raw_block(node.features, node.first, node.begin, node.end)


class CappedForest(Forest384):
    def __init__(self, cap_pairs=CAP_PAIRS):
        super().__init__()
        if type(cap_pairs) is not int or cap_pairs < 1:
            raise ValueError('positive fixed cap required')
        self.cap_pairs = cap_pairs
        self.forced_seals = 0

    def append(self, block, *, starts_event):
        n = self.legacy.raw(block)
        if (type(starts_event) is not bool or n.first != self.last_index + 1
                or (self.last_end is not None and n.begin <= self.last_end)
                or (self.width is not None and n.features.shape[1] != self.width)):
            raise ValueError('consecutive causal pair with fixed width required')
        if len(self.recent) == 4:
            old, boundary = self.recent.popleft()
            forced = (self.open is not None
                      and self.open.last - self.open.first + 1 >= self.cap_pairs)
            if (boundary or forced) and self.open is not None:
                self.roots.append(self.open)
                self.open = None
                self.forced_seals += int(forced and not boundary)
            if self.open is None:
                self.open = old
            else:
                self.open = self.legacy.merge(self.open, old, within=True)
                self.within_merges += 1
            while sum(len(x.features) for x in self.roots) > 144:
                costs = self.legacy.penalties(self.roots, old.end)
                i = min(range(len(costs)), key=lambda j: costs[j]['total'])
                self.roots[i:i+2] = [self.legacy.merge(self.roots[i], self.roots[i+1])]
                self.long_merges += 1
        self.recent.append((n, starts_event))
        self.last_index, self.last_end, self.width = n.last, n.end, n.features.shape[1]
        state = self.status()
        if (state['tokens'] > 384 or state['closed_tokens'] > 144
                or state['metadata_bytes'] > 32768 or state['open_pair_span'] > self.cap_pairs):
            raise RuntimeError('candidate storage bound exceeded')

    def status(self):
        state = super().status()
        state.update(cap_pairs=self.cap_pairs, forced_seals=self.forced_seals,
                     open_pair_span=0 if self.open is None else self.open.last-self.open.first+1,
                     closed_nodes_can_still_merge=True)
        return state


class ReservoirRecent:
    """Algorithm R over promoted old pairs; four complete recent pairs stay intact.

    One fixed RNG seed per video. No full-history container, question or future length.
    A single draw is a simple stochastic baseline, not a deterministic uniform grid.
    """
    def __init__(self, seed=RESERVOIR_SEED):
        self.recent = deque()
        self.old = []
        self.promoted = 0
        self.last = -1
        self.rng = random.Random(seed)

    def append(self, node):
        if node.first != self.last + 1 or not node.metadata()['exact_pair']:
            raise ValueError('complete sequential pairs required')
        if len(self.recent) == 4:
            item = self.recent.popleft()
            self.promoted += 1
            if len(self.old) < 4:
                self.old.append(item)
            else:
                j = self.rng.randrange(self.promoted)
                if j < 4:
                    self.old[j] = item
        self.recent.append(node.copy())
        self.last = node.last

    def export(self):
        return [n.copy() for n in sorted([*self.old, *self.recent], key=lambda n: n.first)]

    def status(self):
        ns = sorted([*self.old, *self.recent], key=lambda n: n.first)
        return dict(tokens=sum(len(n.features) for n in ns),
                    tensor_bytes=sum(n.metadata()['tensor_bytes'] for n in ns),
                    retained_pairs=[n.first for n in ns], promoted=self.promoted,
                    capacity_pairs=8, old_capacity=4, recent_capacity=4,
                    metadata_payloads=len(ns), fixed_seed=RESERVOIR_SEED)
