"""LS shared history, with an optional depth-first adjacent overflow selector.

Adapted from wshy531/qwen35-streaming-2b, shared.py at
4212d2399954f19700b568cc806ed82fece3349f. Original mode is checked
step by step against the unchanged snapshot; only the selector differs.
"""
from collections import deque
from dataclasses import dataclass
import json
import torch

from common import ROOT
from legacy_sources import load


@dataclass(frozen=True)
class EventNode:
    node: object
    first_event: int
    last_event: int


class SharedHistory384:
    def __init__(self, *, depth_first=False):
        self.legacy = load().forest
        self.depth_first = depth_first
        self.recent = deque()
        self.history = []
        self.current_event = -1
        self.last_index, self.last_end, self.width = -1, None, None
        self.within_merges = self.long_merges = 0

    def nodes(self):
        return [entry.node for entry in (*self.history, *self.recent)]

    def controls(self):
        return dict(last_index=self.last_index, last_end=self.last_end, width=self.width,
            current_event=self.current_event, within_merges=self.within_merges,
            long_merges=self.long_merges,
            history_events=[[e.first_event, e.last_event] for e in self.history],
            recent_events=[[e.first_event, e.last_event] for e in self.recent])

    def status(self):
        nodes = [n.metadata() for n in self.nodes()]
        return dict(tokens=sum(n['tokens'] for n in nodes),
            history_tokens=sum(len(e.node.features) for e in self.history),
            recent_tokens=sum(len(e.node.features) for e in self.recent),
            uncompressed_history_pairs=sum(e.node.metadata()['exact_pair'] for e in self.history),
            nodes=nodes, resident_tensor_bytes=sum(n['tensor_bytes'] for n in nodes),
            metadata_bytes=len(json.dumps(dict(nodes=nodes, controls=self.controls())).encode()),
            within_merges=self.within_merges, long_merges=self.long_merges)

    def select(self, costs):
        if not self.depth_first:
            return min(range(len(costs)), key=lambda i: (costs[i]['total'], i))
        return min(range(len(costs)), key=lambda i: (
            1 + max(self.history[i].node.depth, self.history[i+1].node.depth),
            costs[i]['total'], i))

    @torch.no_grad()
    def append(self, block, *, starts_event):
        node = self.legacy.raw(block)
        if (type(starts_event) is not bool or node.first != self.last_index+1
                or (self.last_end is not None and node.begin <= self.last_end)
                or (self.width is not None and node.features.shape[1] != self.width)):
            raise ValueError('consecutive pair, fixed width and explicit causal event flag required')
        event = 0 if self.last_index == -1 else self.current_event+int(starts_event)
        if len(self.recent) == 4:
            old = self.recent.popleft()
            self.history.append(old)
            while sum(len(e.node.features) for e in self.history) > 192:
                costs = self.legacy.penalties([e.node for e in self.history], old.node.end)
                index = self.select(costs)
                left, right = self.history[index:index+2]
                within = left.first_event == left.last_event == right.first_event == right.last_event
                merged = self.legacy.merge(left.node, right.node, within=within)
                self.history[index:index+2] = [EventNode(merged, left.first_event, right.last_event)]
                if within:
                    self.within_merges += 1
                else:
                    self.long_merges += 1
        self.recent.append(EventNode(node, event, event))
        self.current_event = event
        self.last_index, self.last_end, self.width = node.last, node.end, node.features.shape[1]
        state = self.status()
        if state['tokens'] > 384 or state['history_tokens'] > 192 or state['metadata_bytes'] > 32768:
            raise RuntimeError('shared history storage bound exceeded')

    def export(self):
        return [n.copy() for n in self.nodes()]
