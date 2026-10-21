"""Frozen capacity policies; Algorithm R reused at12 old +4 recent pairs."""
from collections import deque
from pathlib import Path
import random,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-spatial-time-v1'))
from policy import FixedForest,RecentBank,raw_block


class Reservoir768:
    def __init__(self):
        self.recent=deque();self.old=[];self.promoted=0;self.last=-1;self.last_end=None
        self.rng=random.Random(20261020)

    def append(self,block):
        from forest import raw
        node=raw(block)
        if node.first!=self.last+1 or not node.metadata()['exact_pair'] or (
            self.last_end is not None and node.begin<=self.last_end):
            raise ValueError('next complete causal pair required')
        if len(self.recent)==4:
            item=self.recent.popleft();self.promoted+=1
            if len(self.old)<12:self.old.append(item)
            else:
                j=self.rng.randrange(self.promoted)
                if j<12:self.old[j]=item
        self.recent.append(node);self.last=node.last;self.last_end=node.end

    def nodes(self):return sorted([*self.old,*self.recent],key=lambda n:n.first)
    def export(self):return [n.copy() for n in self.nodes()]
    def status(self):
        nodes=self.nodes()
        return dict(tokens=sum(len(n.features) for n in nodes),resident_tensor_bytes=sum(n.metadata()['tensor_bytes'] for n in nodes),
            retained_pairs=[n.first for n in nodes],promoted=self.promoted,old_capacity=12,recent_capacity=4,seed=20261020)
