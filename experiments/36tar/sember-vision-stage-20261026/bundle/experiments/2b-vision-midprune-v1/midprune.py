"""One fixed, frame-local pruning point. Original weights, coordinates and merger stay intact."""
import torch


@torch.inference_mode()
def encode(visual,pixels,grid_thw,*,mode='mid_odd',return_trace=False):
    if mode not in ('full_replay','post_odd','mid_odd'):
        raise ValueError('fixed registered pruning mode required')
    if visual.training or next(visual.parameters()).dtype!=torch.float32:
        raise ValueError('FP32 eval visual encoder required')
    if grid_thw.shape!=(1,3) or int(grid_thw[0,0])!=1 or visual.spatial_merge_size!=2:
        raise ValueError('one real temporal pair with original 2x2 merger required')
    h,w=[int(v) for v in grid_thw[0,1:]]
    if (h,w) not in ((12,16),(18,24)) or len(visual.blocks)<2 or len(visual.blocks)%2:
        raise ValueError('registered pair grid and even-depth encoder required')
    cut=len(visual.blocks)//2
    count=h*w//4
    groups=torch.arange(1,count,2,device=pixels.device)
    patches=(groups[:,None]*4+torch.arange(4,device=pixels.device)[None,:]).flatten()
    x=visual.patch_embed(pixels)
    x=x+visual.fast_pos_embed_interpolate(grid_thw)
    rope=visual.rot_pos_emb(grid_thw)
    length=x.shape[0]
    rope=rope.reshape(length,-1)
    emb=torch.cat((rope,rope),dim=-1)
    positions=(emb.cos(),emb.sin())
    cu=torch.tensor([0,length],dtype=torch.int32,device=pixels.device)
    trace=[]
    for i,block in enumerate(visual.blocks):
        if mode=='mid_odd' and i==cut:
            x=x.index_select(0,patches)
            positions=tuple(t.index_select(0,patches) for t in positions)
            cu=torch.tensor([0,len(patches)],dtype=torch.int32,device=pixels.device)
        if return_trace:trace.append(int(x.shape[0]))
        x=block(x,cu_seqlens=cu,position_embeddings=positions)
    features=visual.merger(x)
    if mode=='post_odd':features=features.index_select(0,groups)
    if mode=='full_replay':groups=torch.arange(count,device=pixels.device)
    if features.shape[0]!=len(groups):raise AssertionError('merger output and original group IDs disagree')
    return features,groups,dict(layer_tokens=trace,cut_after_layers=cut,
        original_grid=[h//2,w//2],original_patch_count=length,retained_patches=int(x.shape[0]),
        returned_groups=len(groups),cross_frame_feature_cache=False)
