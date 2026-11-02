"""Shared whole-node scorer; frozen VLM representations are inputs, not weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class EvidenceSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(2048,64)
        self.visual=nn.Linear(4096,64)
        self.scorer=nn.Sequential(nn.Linear(328,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,query,recent,history,metadata,mask,*,blind_history=False):
        if query.ndim!=2 or query.shape[-1]!=2048 or recent.shape!=(len(query),4096):
            raise ValueError('frozen question/current descriptors required')
        b,n,d=history.shape
        if b!=len(query) or d!=4096 or metadata.shape!=(b,n,8) or mask.shape!=(b,n):
            raise ValueError('bounded candidate shape mismatch')
        if n>4 or mask.dtype!=torch.bool:raise ValueError('at most four bank-history nodes')
        q=F.layer_norm(self.query(query),(64,));r=F.layer_norm(self.visual(recent),(64,))
        values=torch.zeros_like(history) if blind_history else history
        h=F.layer_norm(self.visual(values),(64,))
        # NULL is a reading decision, not a guarantee of factual answerability.
        null_meta=torch.zeros((b,1,8),device=query.device,dtype=query.dtype);null_meta[:,:,-1]=1
        h=torch.cat((r[:,None,:],h),1);meta=torch.cat((null_meta,metadata),1)
        qr=q[:,None,:].expand(-1,n+1,-1);rr=r[:,None,:].expand_as(qr)
        logits=self.scorer(torch.cat((qr,rr,h,qr*h,rr*h,meta),-1)).squeeze(-1)
        logits[:,1:]=logits[:,1:].masked_fill(~mask,-torch.inf)
        counts=mask.sum(1);has=counts>0
        safe=torch.where(has[:,None],logits[:,1:],torch.zeros_like(logits[:,1:]))
        history_mean=torch.where(has,torch.logsumexp(safe,1)-counts.clamp_min(1).log(),torch.full_like(logits[:,0],-torch.inf))
        return logits,history_mean-logits[:,0]

def objective(model,batch,*,blind_history=False):
    logits,route=model(batch['query'],batch['recent'],batch['history'],batch['metadata'],batch['mask'],blind_history=blind_history)
    known=batch['route_known'];has=batch['mask'].any(1)
    if bool(torch.any(known & ~has & (batch['needs_history']>0))):raise ValueError('historical target without candidate')
    valid=known & has
    route_loss=logits.new_zeros(())
    if valid.any():
        per=F.binary_cross_entropy_with_logits(route[valid],batch['needs_history'][valid],reduction='none')
        w=batch['route_weight'][valid];route_loss=(per*w).sum()/w.sum().clamp_min(1e-8)
    positive=batch['positive_mask'] & batch['mask'];known=positive.any(1)
    rank_loss=logits.new_zeros(())
    if known.any():
        hist=logits[known,1:];pos=positive[known]
        # Positive-unlabelled citation mass; uncited nodes are not proven negatives.
        loss=torch.logsumexp(hist,1)-torch.logsumexp(hist.masked_fill(~pos,-torch.inf),1)
        w=batch['rank_weight'][known];rank_loss=(loss*w).sum()/w.sum().clamp_min(1e-8)
    return route_loss+.5*rank_loss,dict(route=route_loss,weak_citation_rank=rank_loss)

def select(logits,route_logit,node_tokens):
    if logits.ndim!=1 or len(logits)!=len(node_tokens)+1:raise ValueError('one query and its bank nodes')
    if math.isnan(float(route_logit)) or float(route_logit)==math.inf:raise FloatingPointError('invalid selector score')
    if float(route_logit)==-math.inf:return dict(history_indices=[],reading='recent',evidence_verified=False)
    if float(route_logit)<=0:return dict(history_indices=[],reading='recent',evidence_verified=False)
    order=sorted(range(len(node_tokens)),key=lambda i:(-float(logits[i+1]),i));picked=[];tokens=0
    for i in order:
        if float(logits[i+1])<=float(logits[0]):continue
        if len(picked)<2 and tokens+node_tokens[i]<=432:picked.append(i);tokens+=node_tokens[i]
    return dict(history_indices=sorted(picked),reading='history' if picked else 'recent',
                selected_history_tokens=tokens,evidence_verified=False)

def metadata(node,query_time,recent_begin):
    if not 0<=node['begin']<=node['end']<=recent_begin<=query_time:raise ValueError('causal older candidate required')
    scale=math.log1p(3600)
    return [math.log1p(query_time-node['begin'])/scale,math.log1p(query_time-node['end'])/scale,
        math.log1p(node['end']-node['begin'])/scale,math.log1p(node.get('depth',0))/math.log1p(128),
        node['tokens']/432,float(not node['exact_pair']),math.log1p(recent_begin-node['end'])/scale,0.]
