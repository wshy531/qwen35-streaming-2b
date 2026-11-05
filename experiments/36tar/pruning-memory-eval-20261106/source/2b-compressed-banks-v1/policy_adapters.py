"""Token-budget adapters; original merge/penalties unchanged, no global patch."""
from types import SimpleNamespace
import torch
from memory384 import Forest384
from shared import SharedHistory384,EventNode
from legacy_sources import load
from forest import Node,validate
from features import validate as validate_block

def pair_node(block):
    validate_block(block)
    if block.first!=block.last or len(block.features) not in (24,48) or not bool(torch.all(block.mass==1)):
        raise ValueError('one selected or full original pair required')
    n=Node(block.features.clone(),block.mass.clone(),
        torch.cat((torch.full((len(block.features),1),block.first,dtype=torch.long),block.xy.clone()),1),
        block.first,block.last,block.begin,block.end,block.center)
    validate(n)
    return n

def adapter(pair_tokens):
    original=load().forest
    def checked(block):
        n=pair_node(block)
        if len(n.features)!=pair_tokens:raise ValueError('fixed per-session pair token count required')
        return n
    return SimpleNamespace(raw=checked,merge=original.merge,penalties=original.penalties)

class BoundedForest(Forest384):
    def __init__(self,depth_first=False,pair_tokens=48):
        super().__init__()
        if pair_tokens not in (24,48):raise ValueError('registered pair sizes required')
        self.legacy=adapter(pair_tokens);self.depth_first=depth_first
    def select(self,costs):
        if not self.depth_first:return min(range(len(costs)),key=lambda i:costs[i]['total'])
        return min(range(len(costs)),key=lambda i:(1+max(self.roots[i].depth,self.roots[i+1].depth),costs[i]['total'],i))
    @torch.no_grad()
    def append(self,block,*,starts_event):
        n=self.legacy.raw(block)
        if type(starts_event) is not bool or n.first!=self.last_index+1 or (
            self.last_end is not None and n.begin<=self.last_end) or (
            self.width is not None and n.features.shape[1]!=self.width):
            raise ValueError('consecutive pair, fixed width and explicit causal event flag required')
        if sum(len(n.features) for n,_ in self.recent)>=192:
            old,boundary=self.recent.popleft()
            if boundary and self.open is not None:
                self.roots.append(self.open);self.open=None
            if self.open is None:self.open=old
            else:
                self.open=self.legacy.merge(self.open,old,within=True);self.within_merges+=1
            while sum(len(n.features) for n in self.roots)>144:
                costs=self.legacy.penalties(self.roots,old.end)
                i=self.select(costs)
                self.roots[i:i+2]=[self.legacy.merge(self.roots[i],self.roots[i+1])]
                self.long_merges+=1
        self.recent.append((n,starts_event));self.last_index=n.last;self.last_end=n.end
        self.width=n.features.shape[1]
        state=self.status()
        if state['tokens']>384 or state['closed_tokens']>144 or state['metadata_bytes']>32768:
            raise RuntimeError('same-budget memory bounds exceeded')

class BoundedShared(SharedHistory384):
    def __init__(self,pair_tokens=48):
        super().__init__()
        if pair_tokens not in (24,48):raise ValueError('registered pair sizes required')
        self.legacy=adapter(pair_tokens)
    @torch.no_grad()
    def append(self, block, *, starts_event):
        node = self.legacy.raw(block)
        if (type(starts_event) is not bool or node.first != self.last_index+1
                or (self.last_end is not None and node.begin <= self.last_end)
                or (self.width is not None and node.features.shape[1] != self.width)):
            raise ValueError('consecutive pair, fixed width and explicit causal event flag required')
        event = 0 if self.last_index == -1 else self.current_event+int(starts_event)
        if sum(len(e.node.features) for e in self.recent) >= 192:
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
