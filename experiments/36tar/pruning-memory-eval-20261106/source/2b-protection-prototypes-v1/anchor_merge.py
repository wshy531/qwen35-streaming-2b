"""ToMe-inspired fixed-anchor LOCAL group merge, not original ToMe.

Keep original selected IDs; merge four within-group slots separately.
No future features, no question access, no learned parameters. No mass attention
bias is added: that missing adjustment and inherited coordinates require QA validation.
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def merge_groups(x,ids,grid=(9,12)):
    h,w=grid;n=h*w
    if x.ndim!=2 or x.shape[0]!=4*n or not torch.isfinite(x).all():raise ValueError('finite full group features')
    if ids.ndim!=1 or ids.dtype!=torch.long or ids.device!=x.device or not len(ids):raise ValueError('anchor IDs')
    if bool((ids<0).any() or (ids>=n).any()) or bool((ids[1:]<=ids[:-1]).any()):raise ValueError('sorted unique IDs')
    groups=x.reshape(n,4,-1);all_ids=torch.arange(n,device=x.device)
    coords=torch.stack((all_ids//w,all_ids%w),1)
    distance=(coords[:,None]-coords[ids][None]).abs().amax(-1)
    eligible=distance<=1
    if not bool(eligible.any(1).all()):raise ValueError('every source must have a local anchor')
    z=F.normalize(groups.mean(1).float(),dim=-1)
    sim=(z@z[ids].T).masked_fill(~eligible,-torch.inf)
    assignment=sim.argmax(1);assignment[ids]=torch.arange(len(ids),device=x.device)
    mass=torch.bincount(assignment,minlength=len(ids))
    merged=torch.zeros_like(groups[ids]);merged.index_add_(0,assignment,groups)
    merged=merged/mass[:,None,None].to(merged)
    return merged.reshape(-1,x.shape[-1]),dict(assignment=assignment,mass=mass,ids=ids.clone())
