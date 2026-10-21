"""Optional A06 closed-root depth selector, ported from the locked teammate source.

Upstream policy.py SHA256:
4404e3feae920f0d26f4296718274dda328c8ff41a8d82287d636617fab28bae
Only the selector differs from frozen M2. No observation/language core is used.
"""
import torch

from .frozen import policies


class DepthForest384(policies().forest384.Forest384):
    def __init__(self):
        super().__init__()
        self.selection_count = 0

    def select(self, costs):
        return min(range(len(costs)), key=lambda i: (
            1 + max(self.roots[i].depth, self.roots[i+1].depth), costs[i]['total'], i))

    @torch.no_grad()
    def append(self, block, *, starts_event):
        # Keep upstream validation, delayed event promotion and numeric merge.
        n = self.legacy.raw(block)
        if type(starts_event) is not bool or n.first != self.last_index+1 or (
            self.last_end is not None and n.begin <= self.last_end) or (
            self.width is not None and n.features.shape[1] != self.width):
            raise ValueError('consecutive pair, fixed width and explicit causal event flag required')
        if len(self.recent) == 4:
            old, boundary = self.recent.popleft()
            if boundary and self.open is not None:
                self.roots.append(self.open)
                self.open = None
            if self.open is None:
                self.open = old
            else:
                self.open = self.legacy.merge(self.open, old, within=True)
                self.within_merges += 1
            while sum(len(n.features) for n in self.roots) > 144:
                costs = self.legacy.penalties(self.roots, old.end)
                i = self.select(costs)
                self.roots[i:i+2] = [self.legacy.merge(self.roots[i], self.roots[i+1])]
                self.long_merges += 1
                self.selection_count += 1
        self.recent.append((n, starts_event))
        self.last_index, self.last_end = n.last, n.end
        self.width = n.features.shape[1]
        state = self.status()
        if state['tokens'] > 384 or state['closed_tokens'] > 144 or state['metadata_bytes'] > 32768:
            raise RuntimeError('same-budget memory bounds exceeded')
