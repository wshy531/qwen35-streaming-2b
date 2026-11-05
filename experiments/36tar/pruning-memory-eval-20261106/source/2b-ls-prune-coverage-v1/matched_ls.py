"""Hold the LS event and merge schedule fixed for a pruning diagnostic."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-midprune-memory-v1'))
from adapter import sparse_pair
from resolution_banks_base import Bank


class ScheduledLS(Bank):
    def __init__(self, keep):
        super().__init__('LS', keep, budget=8 * keep, recent=4 * keep, closed=3 * keep)
        self._replay = None
        self.last_choices = []

    def _choose(self, costs):
        choice = super()._choose(costs) if self._replay is None else next(self._replay)
        if not 0 <= choice < len(costs):
            raise ValueError('reference merge choice outside current history')
        self.last_choices.append(choice)
        return choice

    def append_step(self, node, starts_event, choices=None):
        self.last_choices = []
        self._replay = None if choices is None else iter(choices)
        try:
            self.append(node, starts_event=starts_event)
            if choices is not None and self.last_choices != list(choices):
                raise RuntimeError('reference merge choices were not fully replayed')
        finally:
            self._replay = None


def structure(bank):
    return [(e.node.first, e.node.last, e.node.begin, e.node.end, e.node.depth,
             e.first_event, e.last_event) for e in (*bank.history, *bank.recent)]


class MatchedLS:
    """A diagnostic that pays for a full branch to establish the shared schedule."""

    def __init__(self, views):
        self.full = ScheduledLS(108)
        self.pruned = {view: ScheduledLS(54) for view in views}
        self.invalid = False

    def append(self, full_node, pruned_nodes, *, starts_event):
        if self.invalid or self.pruned.keys() != pruned_nodes.keys():
            raise ValueError('complete set of registered views required')
        try:
            self.full.append_step(full_node, starts_event)
            expected = structure(self.full)
            for view, bank in self.pruned.items():
                bank.append_step(pruned_nodes[view], starts_event, tuple(self.full.last_choices))
                if structure(bank) != expected:
                    raise RuntimeError(f'{view} failed to match full temporal structure')
        except BaseException:
            self.invalid = True
            raise
        return tuple(self.full.last_choices)
