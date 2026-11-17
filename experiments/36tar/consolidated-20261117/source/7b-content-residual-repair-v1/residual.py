"""Candidate: exact content-selection start, train only a zero-output correction.

Same fixed 70-group budget and mandatory one-per-tile coverage as CoverageScorer.
Not enabled in the existing trainer or default inference path.
"""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-trained-prune-transfer-v1'))
import hard_prune

class ContentResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.correction=hard_prune.GroupScorer(width)
        nn.init.zeros_(self.correction.net[-1].weight)
        nn.init.zeros_(self.correction.net[-1].bias)

    def forward(self,x,previous=None):
        z=x.reshape(140,4,-1).mean(1).float()
        spatial=1-F.cosine_similarity(z,z.mean(0)[None],dim=-1,eps=1e-12)
        if previous is None:base=spatial
        else:
            assert previous.shape==z.shape
            base=.5*spatial+.5*(1-F.cosine_similarity(z,previous,dim=-1,eps=1e-12))
        delta,state=self.correction(x,previous)
        return base.detach()+delta,state

def select(scores):
    assert scores.shape==(140,) and torch.isfinite(scores).all()
    rank=torch.argsort(scores.detach(),descending=True,stable=True)
    ids=torch.arange(140,device=scores.device)
    tile=(ids//14//2)*7+(ids%14//2)
    priority=torch.empty_like(rank);priority[rank]=ids
    winners=torch.full((35,),140,device=scores.device,dtype=torch.long)
    winners.scatter_reduce_(0,tile,priority,reduce='amin',include_self=True)
    keep=torch.zeros(140,device=scores.device,dtype=torch.bool);keep[rank[winners]]=True
    remaining=rank[~keep[rank]];keep[remaining[:35]]=True
    chosen=ids[keep];assert len(chosen)==70 and len(torch.unique(tile[chosen]))==35
    return chosen

def encode(visual,pixels,grid,scorer,previous=None,mode='gather'):
    state=hard_prune.prefix(visual,pixels,grid)
    x,_,order,_=state
    original=x.reshape(140,4,-1)[torch.argsort(order)].reshape(560,-1)
    scores,past=scorer(original,previous);ids=select(scores)
    return hard_prune.tail(visual,state,scores,ids,mode),ids,scores,past
