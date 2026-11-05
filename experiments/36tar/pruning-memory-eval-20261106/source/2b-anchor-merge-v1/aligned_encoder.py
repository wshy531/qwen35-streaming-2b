"""Match deployed heuristic11 exactly; only replace deletion with local merge."""
import sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-protection-prototypes-v1'))
from anchor_merge import merge_groups

@torch.inference_mode()
def encode(visual,pixels,grid,scorer,*,merge=True):
    if visual.training or next(visual.parameters()).dtype!=torch.float32:raise ValueError('FP32 eval only')
    if tuple(grid.shape)!=(1,3) or grid[0].tolist()!=[1,18,24] or len(visual.blocks)!=24:raise ValueError('fixed original visual layout/depth')
    x=visual.patch_embed(pixels)+visual.fast_pos_embed_interpolate(grid)
    rope=visual.rot_pos_emb(grid).reshape(len(x),-1);emb=torch.cat((rope,rope),-1)
    positions=(emb.cos(),emb.sin());cu=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
    for block in visual.blocks[:11]:x=block(x,cu_seqlens=cu,position_embeddings=positions)
    ids=scorer.choose(x,(9,12))
    x=visual.blocks[11](x,cu_seqlens=cu,position_embeddings=positions)
    patches=(ids[:,None]*4+torch.arange(4,device=x.device)).flatten()
    meta={}
    if merge:x,meta=merge_groups(x,ids)
    else:x=x[patches]
    positions=tuple(p[patches] for p in positions)
    cu=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
    for block in visual.blocks[12:]:x=block(x,cu_seqlens=cu,position_embeddings=positions)
    return visual.merger(x),ids,meta
