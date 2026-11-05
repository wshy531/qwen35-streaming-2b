"""Causal fixed-quota projection selection; DTD-inspired, not a DTD reproduction."""
import torch
import torch.nn.functional as F

def odd_indices():return torch.arange(1,48,2)

def spatial_anchors():
    # Half of the odd24 lattice: two sites per row, alternate between odd columns.
    return torch.tensor([i for i in range(48) if i%2==1 and (i//8+(i%8)//2)%2==0])

@torch.inference_mode()
def select(features, previous=None):
    if features.ndim!=2 or len(features)!=48 or features.device.type!='cpu' or not torch.isfinite(features).all():
        raise ValueError('finite complete CPU pair required')
    odd=odd_indices()
    if previous is None:
        return dict(odd24=odd,change24=odd.clone(),mix24=odd.clone()),None
    if previous.shape!=features.shape or previous.device.type!='cpu' or not torch.isfinite(previous).all():
        raise ValueError('matching finite previous pair required')
    score=(1-(F.normalize(features.float(),dim=-1,eps=1e-12)*
              F.normalize(previous.float(),dim=-1,eps=1e-12)).sum(-1)).clamp(0,2)
    both_zero=(features.norm(dim=-1)<=1e-12)&(previous.norm(dim=-1)<=1e-12)
    score[both_zero]=0
    ranked=torch.argsort(score,descending=True,stable=True)
    anchors=spatial_anchors()
    extra=ranked[~torch.isin(ranked,anchors)][:12]
    return dict(odd24=odd,change24=ranked[:24].sort().values,
                mix24=torch.cat((anchors,extra)).sort().values),score
