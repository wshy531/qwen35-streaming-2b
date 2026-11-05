"""Predict incremental gold-option probability from frozen bank descriptors.

This is supervised CPU regression on measured TRAINING combinations, not
backpropagation through top-k or through the answering model. No gold answer
or measured utility is supplied to forward().
"""
import itertools
import torch
from torch import nn
from torch.nn import functional as F

def subsets(n):
    if not 0<=n<=4:raise ValueError('requires at most four original history nodes')
    return [tuple(x) for size in range(n+1) for x in itertools.combinations(range(n),size)]

def probability_gain(option_logits,gold_indices):
    """[questions, subsets, rotations, 5]; subset 0 must be recent-only.

    Gold index changes with option rotation. The same question must keep the
    same subsets/visual bank across rotations. Output is a teacher TARGET.
    """
    if option_logits.ndim!=4 or option_logits.shape[-1]!=5:raise ValueError('expected Q,S,R,5')
    q,s,r,_=option_logits.shape
    if gold_indices.shape!=(q,r) or not torch.isfinite(option_logits).all():raise ValueError('invalid targets or logits')
    if bool(((gold_indices<0)|(gold_indices>4)).any()):raise ValueError('gold out of range')
    p=option_logits.detach().float().softmax(-1)
    correct=p.gather(-1,gold_indices[:,None,:,None].expand(q,s,r,1)).squeeze(-1)
    per_rotation=correct-correct[:,0:1,:]
    return per_rotation.mean(-1),dict(correct_probability=correct,per_rotation_gain=per_rotation)

class HistoryUtilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(2048,64)
        self.visual=nn.Linear(4096,64)
        self.score=nn.Sequential(nn.Linear(393,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,query,recent,history,metadata,node_mask,subset_mask,*,blind_history=False):
        b,n,d=history.shape
        if n!=4 or d!=4096 or query.shape!=(b,2048) or recent.shape!=(b,4096):raise ValueError('fixed descriptor dimensions')
        if metadata.shape!=(b,4,8) or node_mask.shape!=(b,4) or subset_mask.ndim!=3 or subset_mask.shape[0]!=b or subset_mask.shape[2]!=4:raise ValueError('fixed bank shape')
        if node_mask.dtype!=torch.bool or subset_mask.dtype!=torch.bool:raise ValueError('boolean masks')
        if bool((subset_mask & ~node_mask[:,None,:]).any()):raise ValueError('subset contains absent node')
        if bool(subset_mask[:,0].any()):raise ValueError('candidate zero must be NULL')
        q=F.layer_norm(self.query(query),(64,));r=F.layer_norm(self.visual(recent),(64,))
        raw=torch.zeros_like(history) if blind_history else history
        h=F.layer_norm(self.visual(raw),(64,));count=subset_mask.sum(-1,keepdim=True)
        weights=subset_mask.to(h.dtype);den=count.clamp_min(1)
        mean=torch.einsum('bsn,bnd->bsd',weights,h)/den
        expanded=h[:,None].expand(-1,subset_mask.shape[1],-1,-1)
        maximum=expanded.masked_fill(~subset_mask[:,:,:,None],-torch.inf).max(2).values
        maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        meta=torch.einsum('bsn,bnd->bsd',weights,metadata)/den
        qq=q[:,None,:].expand_as(mean);rr=r[:,None,:].expand_as(mean)
        values=self.score(torch.cat((qq,rr,mean,maximum,qq*mean,rr*mean,meta,count/4),-1)).squeeze(-1)
        # Relative model prediction; NULL is exactly zero by construction.
        return torch.tanh(values-values[:,0:1])

def objective(prediction,target,candidate_mask,source_weights):
    if prediction.shape!=target.shape or candidate_mask.shape!=target.shape:raise ValueError('candidate target shape')
    valid=candidate_mask.clone();valid[:,0]=False
    per=F.huber_loss(prediction,target.detach(),delta=.1,reduction='none')
    per=(per*valid).sum(1)/valid.sum(1).clamp_min(1)
    # Caller normalizes inverse-source-frequency weights ONCE over training
    # rows. Do not renormalize each minibatch and erase source balancing.
    return (per*source_weights).mean()

def choose(predicted,choices):
    if predicted.ndim!=1 or len(predicted)!=len(choices) or not torch.isfinite(predicted).all():raise ValueError('invalid selection scores')
    # One actual set is read. No soft blend of different video features.
    return max(range(len(choices)),key=lambda i:(float(predicted[i]),-len(choices[i]),tuple(-j for j in choices[i])))
