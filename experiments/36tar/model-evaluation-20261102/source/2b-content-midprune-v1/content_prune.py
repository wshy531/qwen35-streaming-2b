"""Causal, fixed-budget whole-merger-group selection. No weights or hidden archive."""
import torch
import torch.nn.functional as F


class CoverageScorer:
    """One winner per 2x2 output-grid tile, then fill remaining half-budget slots.

    score = .5 spatial distinctiveness + .5 same-position temporal change.
    The first pair uses spatial distinctiveness only. Ties prefer lower IDs.
    The previous state contains pooled *middle-layer* groups, never final features.
    """
    def __init__(self):
        self.previous = None
        self.grid = None

    def reset(self):
        self.previous = None
        self.grid = None

    @torch.inference_mode()
    def choose(self, x, grid):
        h, w = grid
        if x.ndim != 2 or len(x) != h*w*4 or not torch.isfinite(x).all():
            raise ValueError('finite full 2x2-merger-group input required')
        z = x.reshape(h*w, 4, -1).mean(1).float()
        spatial = 1-F.cosine_similarity(z, z.mean(0)[None], dim=-1, eps=1e-12)
        if self.previous is None:
            score = spatial
        else:
            if self.grid != grid or self.previous.shape != z.shape or self.previous.device != z.device:
                raise ValueError('reset scorer at source/grid/device change')
            temporal = 1-F.cosine_similarity(z, self.previous, dim=-1, eps=1e-12)
            score = .5*spatial + .5*temporal
        rank = torch.argsort(score, descending=True, stable=True)
        # Reduce independently within spatial tiles, without CPU/GPU scalar transfers.
        ids = torch.arange(h*w, device=x.device)
        tile = (ids//w//2)*((w+1)//2)+(ids%w//2)
        priorities = torch.empty_like(rank)
        priorities[rank] = torch.arange(h*w, device=x.device)
        winners = torch.full((((h+1)//2)*((w+1)//2),), h*w, device=x.device, dtype=torch.long)
        winners.scatter_reduce_(0, tile, priorities, reduce='amin', include_self=True)
        selected = torch.zeros(h*w, device=x.device, dtype=torch.bool)
        selected[rank[winners]] = True
        remaining = rank[~selected[rank]]
        keep = h*w//2
        selected[remaining[:keep-len(winners)]] = True
        chosen = ids[selected]
        if len(chosen) != keep:
            raise AssertionError('fixed half budget')
        self.previous, self.grid = z.detach().clone(), grid
        return chosen

    def resident_bytes(self):
        return 0 if self.previous is None else self.previous.numel()*self.previous.element_size()


@torch.inference_mode()
def encode(visual, pixels, grid_thw, *, mode, scorer=None, return_trace=False):
    if mode not in ('full', 'post_fixed', 'mid_fixed', 'post_content', 'mid_content'):
        raise ValueError('registered mode required')
    if visual.training or next(visual.parameters()).dtype != torch.float32:
        raise ValueError('FP32 eval encoder required')
    if grid_thw.shape != (1,3) or int(grid_thw[0,0]) != 1 or visual.spatial_merge_size != 2:
        raise ValueError('one temporal pair, original 2x2 merger required')
    h,w = [int(v) for v in grid_thw[0,1:]]
    if (h,w) != (18,24) or len(visual.blocks)%2:
        raise ValueError('fixed 288x384 / even depth required')
    cut = len(visual.blocks)//2
    count = h*w//4
    ids = torch.arange(count,device=pixels.device)
    x = visual.patch_embed(pixels)+visual.fast_pos_embed_interpolate(grid_thw)
    rope = visual.rot_pos_emb(grid_thw).reshape(len(x),-1)
    emb = torch.cat((rope,rope),dim=-1)
    positions = (emb.cos(),emb.sin())
    cu = torch.tensor([0,len(x)],dtype=torch.int32,device=x.device)
    trace = []
    for i,block in enumerate(visual.blocks):
        if i == cut and mode != 'full':
            if mode.endswith('content'):
                if scorer is None:raise ValueError('per-source scorer required')
                ids = scorer.choose(x,(h//2,w//2))
            else:ids = torch.arange(1,count,2,device=x.device)
            if mode.startswith('mid'):
                patches = (ids[:,None]*4+torch.arange(4,device=x.device)[None]).flatten()
                x = x.index_select(0,patches)
                positions = tuple(p.index_select(0,patches) for p in positions)
                cu = torch.tensor([0,len(x)],dtype=torch.int32,device=x.device)
        if return_trace:trace.append(len(x))
        x = block(x,cu_seqlens=cu,position_embeddings=positions)
    features = visual.merger(x)
    if mode.startswith('post'):features = features.index_select(0,ids)
    return features,ids,dict(layer_tokens=trace,cut_after_layers=cut,
        returned_groups=len(ids),selector_bytes=0 if scorer is None else scorer.resident_bytes(),
        original_grid=[h//2,w//2],selection_uses_middle_layer=True)
