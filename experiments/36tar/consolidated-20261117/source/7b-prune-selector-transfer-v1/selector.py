"""Same utility recipe as the 2B head; new dimensions and newly fitted weights."""
import sys,itertools
from common import *
from torch import nn
sys.path.insert(0,str(ROOT/'experiments/2b-history-utility-v1'))
from utility_head import HistoryUtilityHead,choose,objective

class Head(HistoryUtilityHead):
    def __init__(self,width=3584):
        super().__init__();self.width=width
        self.query=nn.Linear(width,64);self.visual=nn.Linear(2*width,64)
    def forward(self,query,recent,history,metadata,node_mask,subset_mask,*,blind_history=False):
        # Preserve the original network after the two learned input projections.
        b,n,d=history.shape
        assert n==4 and d==2*self.width and query.shape==(b,self.width) and recent.shape==(b,2*self.width)
        assert metadata.shape==(b,4,8) and node_mask.shape==(b,4)
        assert not bool((subset_mask & ~node_mask[:,None,:]).any()) and not bool(subset_mask[:,0].any())
        F=torch.nn.functional
        q=F.layer_norm(self.query(query),(64,));r=F.layer_norm(self.visual(recent),(64,))
        h=F.layer_norm(self.visual(torch.zeros_like(history) if blind_history else history),(64,))
        count=subset_mask.sum(-1,keepdim=True);weights=subset_mask.to(h.dtype);den=count.clamp_min(1)
        mean=torch.einsum('bsn,bnd->bsd',weights,h)/den
        maximum=h[:,None].expand(-1,subset_mask.shape[1],-1,-1).masked_fill(~subset_mask[:,:,:,None],-torch.inf).max(2).values
        maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        meta=torch.einsum('bsn,bnd->bsd',weights,metadata)/den
        qq=q[:,None,:].expand_as(mean);rr=r[:,None,:].expand_as(mean)
        values=self.score(torch.cat((qq,rr,mean,maximum,qq*mean,rr*mean,meta,count/4),-1)).squeeze(-1)
        return torch.tanh(values-values[:,0:1])

def choices_mask(n):
    assert 0<=n<=4
    choices=[list(c) for size in range(n+1) for c in itertools.combinations(range(n),size)]
    mask=torch.zeros(1,len(choices),4,dtype=torch.bool)
    for i,c in enumerate(choices):mask[0,i,c]=True
    return choices,mask

def descriptor(nodes):
    x=torch.cat([n.features for n in nodes]);return torch.cat((x.mean(0),x.amax(0)))
def descriptor_features(query,recent,history):
    width=query.numel();hist=torch.zeros(4,width*2);meta=torch.zeros(4,8);mask=torch.zeros(4,dtype=torch.bool)
    now=max(recent[-1].end,1e-6);pairs=recent[-1].last+1
    for i,n in enumerate(history):
        hist[i]=descriptor([n]);mask[i]=True
        meta[i]=torch.tensor([n.begin/now,n.end/now,n.center/now,(n.end-n.begin)/now,(n.last-n.first+1)/pairs,len(n.features)/140,n.depth/max(pairs,1),n.groups/max(pairs,1)])
    return dict(query=query,recent=descriptor(recent),history=hist,metadata=meta,node_mask=mask)

def prompt(q,shift):
    opts=q['options'];r=opts[shift:]+opts[:shift]
    return q['question']+'\n'+'\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(r))+'\nAnswer with the letter of the correct option only.'

@torch.no_grad()
def forward(model,processor,nodes,text,representation=False):
    emb,pos,ids=segmented.context(model,processor,nodes,text,'segmented')
    h=model.model.language_model(inputs_embeds=emb,position_ids=pos,attention_mask=torch.ones(emb.shape[:2],device=emb.device,dtype=torch.long),use_cache=False).last_hidden_state[0,-1]
    return h.float().cpu() if representation else model.lm_head(h).float().cpu()

def option_ids(processor,n):
    rows=[processor.tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(n)]
    assert all(len(x)==1 for x in rows);return [r[0] for r in rows]
