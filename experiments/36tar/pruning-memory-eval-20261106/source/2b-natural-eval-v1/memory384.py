"""Same-token M2 adapter: 4 recent pairs + 48 open + <=144 closed Tokens.

Reuse legacy feature merge/penalties unchanged. Event starts travel with each
pair through the recent queue, so delayed promotion keeps causal boundaries.
This does not supply an event detector and does not load a model.
"""
from collections import deque
import json

import torch

from legacy_sources import load


class Forest384:
    def __init__(self):
        self.legacy=load().forest
        self.recent=deque();self.roots=[];self.open=None
        self.last_index=-1;self.last_end=None;self.width=None
        self.within_merges=0;self.long_merges=0

    def nodes(self):
        return [*self.roots,*([] if self.open is None else [self.open]),*[n for n,_ in self.recent]]

    def status(self):
        nodes=self.nodes();metadata=[n.metadata() for n in nodes]
        return dict(tokens=sum(len(n.features) for n in nodes),
            recent_tokens=sum(len(n.features) for n,_ in self.recent),
            open_tokens=0 if self.open is None else len(self.open.features),
            closed_tokens=sum(len(n.features) for n in self.roots),
            nodes=metadata,last_index=self.last_index,last_end=self.last_end,
            resident_tensor_bytes=sum(x['tensor_bytes'] for x in metadata),
            represented_mass=sum(float(n.mass.sum()) for n in nodes),
            within_merges=self.within_merges,long_merges=self.long_merges,
            metadata_bytes=len(json.dumps(metadata).encode()),
            recent_event_starts=[(n.first,flag) for n,flag in self.recent],
            boundary_rule='required input flag stored with its own pair; no detector supplied',
            provenance_is_not_full_retained_evidence=True)

    @torch.no_grad()
    def append(self,block,*,starts_event):
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
                costs=self.legacy.penalties(self.roots,old.end)
                i=min(range(len(costs)),key=lambda i:costs[i]['total'])
                self.roots[i:i+2]=[self.legacy.merge(self.roots[i],self.roots[i+1])]
                self.long_merges+=1
        self.recent.append((n,starts_event));self.last_index=n.last;self.last_end=n.end
        self.width=n.features.shape[1]
        state=self.status()
        if state['tokens']>384 or state['closed_tokens']>144 or state['metadata_bytes']>32768:
            raise RuntimeError('same-budget memory bounds exceeded')

    def export(self):return [n.copy() for n in self.nodes()]

