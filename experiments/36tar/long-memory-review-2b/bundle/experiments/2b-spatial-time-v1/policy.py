"""One S24 candidate and fixed384/768 capacity controls; no hidden history."""
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from features import raw_block,reduce
from memory384 import Forest384
from memory_policy import Pair,RecentFIFO


class FixedForest(Forest384):
    def __init__(self,*,budget,old_tokens=48):
        if (budget,old_tokens) not in ((384,48),(768,48),(384,24)):
            raise ValueError('only preregistered capacity/space settings allowed')
        super().__init__();self.budget=budget;self.old_tokens=old_tokens
        self.closed_limit=budget-192-old_tokens;self.spatial_reductions=0

    def old_node(self,node):
        if self.old_tokens==48:return node
        assert node.metadata()['exact_pair']
        block=reduce(raw_block(node.features,node.first,node.begin,node.end),'tome')
        coords=torch.cat((torch.full((24,1),node.first,dtype=torch.long),block.xy),1)
        reduced=self.legacy.Node(block.features,block.mass,coords,node.first,node.last,
            node.begin,node.end,node.center,node.groups,node.depth)
        self.legacy.validate(reduced)
        assert len(reduced.features)==24 and float(reduced.mass.sum())==48
        self.spatial_reductions+=1
        return reduced

    @torch.no_grad()
    def append(self,block,*,starts_event):
        n=self.legacy.raw(block)
        if type(starts_event) is not bool or n.first!=self.last_index+1 or (
            self.last_end is not None and n.begin<=self.last_end) or (
            self.width is not None and n.features.shape[1]!=self.width):
            raise ValueError('consecutive complete pair, fixed width, causal flag required')
        if len(self.recent)==4:
            old,boundary=self.recent.popleft();old=self.old_node(old)
            if boundary and self.open is not None:self.roots.append(self.open);self.open=None
            if self.open is None:self.open=old
            else:
                self.open=self.legacy.merge(self.open,old,within=True);self.within_merges+=1
            while sum(len(n.features) for n in self.roots)>self.closed_limit:
                costs=self.legacy.penalties(self.roots,old.end)
                i=min(range(len(costs)),key=lambda i:costs[i]['total'])
                self.roots[i:i+2]=[self.legacy.merge(self.roots[i],self.roots[i+1])];self.long_merges+=1
        self.recent.append((n,starts_event));self.last_index=n.last;self.last_end=n.end;self.width=n.features.shape[1]
        state=self.status()
        if state['tokens']>self.budget or state['closed_tokens']>self.closed_limit or state['metadata_bytes']>32768:
            raise RuntimeError('fixed tensor/metadata budget exceeded')

    def status(self):
        return dict(**super().status(),budget=self.budget,old_node_tokens=self.old_tokens,
            closed_limit=self.closed_limit,spatial_reductions=self.spatial_reductions)


class RecentBank:
    def __init__(self,budget):
        if budget not in (384,768):raise ValueError('fixed384/768 required')
        self.budget=budget;self.bank=RecentFIFO(budget//48);self.last_index=-1;self.width=None;self.last_end=None

    def nodes(self):return [p.payload for p in self.bank.read()]

    def append(self,block):
        from forest import raw
        node=raw(block)
        if node.first!=self.last_index+1 or (self.width is not None and node.features.shape[1]!=self.width) or (
            self.last_end is not None and node.begin<=self.last_end):raise ValueError('one next causal pair required')
        self.bank.append(Pair(node.first,node.begin,node.end,node))
        self.last_index=node.last;self.width=node.features.shape[1];self.last_end=node.end

    def export(self):return [n.copy() for n in self.nodes()]

    def status(self):
        metadata=[n.metadata() for n in self.nodes()]
        return dict(budget=self.budget,tokens=sum(n['tokens'] for n in metadata),nodes=metadata,
            resident_tensor_bytes=sum(n['tensor_bytes'] for n in metadata))
