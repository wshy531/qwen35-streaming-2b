"""A06 first version: closed-node depth priority, original tie cost unchanged.

Persistent payload remains exactly four recent pairs, one open48 node and
at most three closed48 nodes. No child archive or delayed raw features.
"""
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from memory384 import Forest384


class DepthForest384(Forest384):
    def __init__(self,*,original_selector=False):
        super().__init__()
        self.original_selector=original_selector
        self.selection_count=0

    def select(self,costs):
        if self.original_selector:return min(range(len(costs)),key=lambda i:costs[i]['total'])
        return min(range(len(costs)),key=lambda i:(
            1+max(self.roots[i].depth,self.roots[i+1].depth),costs[i]['total'],i))

    @torch.no_grad()
    def append(self,block,*,starts_event):
        # Match the frozen adapter's validation and promotion exactly. Only the
        # selection when closed roots overflow differs in the candidate mode.
        n=self.legacy.raw(block)
        if type(starts_event) is not bool or n.first!=self.last_index+1 or (
            self.last_end is not None and n.begin<=self.last_end) or (
            self.width is not None and n.features.shape[1]!=self.width):
            raise ValueError('consecutive pair, fixed width and explicit causal event flag required')
        if len(self.recent)==4:
            old,boundary=self.recent.popleft()
            if boundary and self.open is not None:
                self.roots.append(self.open);self.open=None
            if self.open is None:self.open=old
            else:
                self.open=self.legacy.merge(self.open,old,within=True);self.within_merges+=1
            while sum(len(n.features) for n in self.roots)>144:
                costs=self.legacy.penalties(self.roots,old.end);i=self.select(costs)
                self.roots[i:i+2]=[self.legacy.merge(self.roots[i],self.roots[i+1])]
                self.long_merges+=1;self.selection_count+=1
        self.recent.append((n,starts_event));self.last_index=n.last;self.last_end=n.end
        self.width=n.features.shape[1]
        state=self.status()
        if state['tokens']>384 or state['closed_tokens']>144 or state['metadata_bytes']>32768:
            raise RuntimeError('same-budget memory bounds exceeded')
