"""Qwen2.5-VL whole-merger-group pruning, preserving local-window membership.

No installed transformer code is modified. Half-depth is 16/32 layers. The
scorer is the existing causal CoverageScorer, applied in ORIGINAL spatial order.
"""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-content-midprune-v1'))
from content_prune import CoverageScorer

@torch.no_grad()
def encode(visual,pixels,grid_thw,mode='full',scorer=None,trace=False,prefix=None,capture_prefix=False):
    assert mode in ('full','fixed','content','post_fixed','post_content')
    assert not visual.training and visual.spatial_merge_unit==4
    assert grid_thw.tolist()==[[1,20,28]]
    if prefix is None:
        x=visual.patch_embed(pixels)
        rope=visual.rot_pos_emb(grid_thw)
        order,cu_list=visual.get_window_index(grid_thw)
        cu_window=torch.unique_consecutive(torch.tensor(cu_list,device=x.device,dtype=torch.int32))
        order=order.to(x.device)
        x=x.reshape(140,4,-1)[order].reshape(560,-1)
        rope=rope.reshape(140,4,-1)[order].reshape(560,-1)
        emb=torch.cat((rope,rope),-1);pos=(emb.cos(),emb.sin())
        cu_full=torch.tensor([0,560],device=x.device,dtype=torch.int32)
        group_window=torch.repeat_interleave(torch.arange(len(cu_window)-1,device=x.device),(cu_window[1:]-cu_window[:-1])//4)
    else:
        x,pos,order,cu_window,cu_full,group_window=prefix
    current_order=order;selected=torch.arange(140,device=x.device);counts=[]
    cut=len(visual.blocks)//2
    saved_prefix=None
    for i,block in enumerate(visual.blocks):
        if prefix is not None and i<cut:continue
        if i==cut and capture_prefix:saved_prefix=(x,pos,order,cu_window,cu_full,group_window)
        if i==cut and mode!='full':
            original=x.reshape(140,4,-1)[torch.argsort(order)].reshape(560,-1)
            selected=(scorer.choose(original,(10,14)) if 'content' in mode else torch.arange(1,140,2,device=x.device))
            assert len(selected)==70 and torch.equal(selected,selected.sort().values)
            if not mode.startswith('post_'):
                keep=torch.isin(order,selected)
                patch_keep=keep.repeat_interleave(4)
                x=x[patch_keep];pos=tuple(p[patch_keep] for p in pos)
                current_order=order[keep]
                lengths=torch.bincount(group_window[keep],minlength=len(cu_window)-1)*4
                # Empty windows contribute no tokens; drop zero-length segments.
                cu_window=torch.cat((torch.zeros(1,device=x.device,dtype=torch.int32),lengths[lengths>0].cumsum(0).to(torch.int32)))
                cu_full=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
                assert int(cu_window[-1])==len(x)==280
        cu=cu_full if i in visual.fullatt_block_indexes else cu_window
        if trace:counts.append(dict(layer=i,tokens=len(x),cu=cu.cpu().tolist(),global_attention=i in visual.fullatt_block_indexes))
        x=block(x,cu_seqlens=cu,position_embeddings=pos)
    value=visual.merger(x)[torch.argsort(current_order)]
    if mode.startswith('post_'):value=value[selected]
    assert value.shape[0]==len(selected) and torch.isfinite(value).all()
    meta=dict(cut=cut,layers=counts,selector_bytes=0 if scorer is None else scorer.resident_bytes())
    if capture_prefix:meta['prefix']=saved_prefix
    return value,selected,meta
