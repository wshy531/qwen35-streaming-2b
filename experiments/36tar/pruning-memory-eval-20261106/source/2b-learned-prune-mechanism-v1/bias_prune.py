"""STTS-inspired gradient bridge, single real frame pair. Experimental, not a new method claim."""
import sys
import torch
from torch import nn
from torch.nn import functional as F

class GroupScorer(nn.Module):
    def __init__(self,width,hidden=32):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,x,previous=None):
        groups=x.reshape(-1,4,x.shape[-1]).mean(1)
        past=torch.zeros_like(groups) if previous is None else previous.detach()
        if past.shape!=groups.shape:raise ValueError('previous pair grid mismatch')
        return self.net(torch.cat((groups,past),-1)).squeeze(-1),groups.detach().clone()

def biased_block(block,x,positions,group_logits):
    """Functional attention override; no module monkey-patch or future-frame packing."""
    attn=block.attn;n=len(x)
    if n!=len(group_logits)*4:raise ValueError('whole groups only')
    apply_rope=sys.modules[type(attn).__module__].apply_rotary_pos_emb_vision
    q,k,v=attn.qkv(block.norm1(x)).reshape(n,3,attn.num_heads,-1).permute(1,0,2,3).unbind(0)
    q,k=apply_rope(q,k,*positions)
    q,k,v=[z.transpose(0,1).unsqueeze(0) for z in (q,k,v)]
    bias=F.logsigmoid(group_logits).repeat_interleave(4)[None,None,None,:]
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.,is_causal=False,scale=attn.scaling)
    x=x+attn.proj(y.transpose(1,2).reshape(n,-1))
    return x+block.mlp(block.norm2(x))

def encode(visual,pixels,grid,scorer,*,previous=None,keep=54,cut=None):
    if visual.training:raise ValueError('frozen eval backbone required')
    if tuple(grid.shape)!=(1,3) or grid.tolist()[0]!=[1,18,24]:raise ValueError('one fixed 288x384 pair only')
    cut=len(visual.blocks)//2 if cut is None else cut
    if not 1<=cut<len(visual.blocks) or not 1<=keep<=108:raise ValueError('invalid budget or cut')
    with torch.no_grad():
        x=visual.patch_embed(pixels)+visual.fast_pos_embed_interpolate(grid)
        rope=visual.rot_pos_emb(grid).reshape(len(x),-1);emb=torch.cat((rope,rope),-1)
        positions=(emb.cos(),emb.sin());cu=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
        for block in visual.blocks[:cut-1]:x=block(x,cu_seqlens=cu,position_embeddings=positions)
    logits,state=scorer(x,previous)
    x=biased_block(visual.blocks[cut-1],x,positions,logits)
    ids=torch.argsort(logits,descending=True,stable=True)[:keep].sort().values
    patches=(ids[:,None]*4+torch.arange(4,device=x.device)).flatten()
    x=x[patches];positions=tuple(p[patches] for p in positions)
    cu=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
    for block in visual.blocks[cut:]:x=block(x,cu_seqlens=cu,position_embeddings=positions)
    return visual.merger(x),ids,logits,state
