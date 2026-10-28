# Isolated derivative; source and exact changes recorded in adapter_derivation.json.
"""Grid-aware derivative of the frozen feature forest; numerical merge unchanged."""
from dataclasses import dataclass,replace
import hashlib
import json
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Node:
    features:torch.Tensor
    mass:torch.Tensor
    coords:torch.Tensor
    first:int
    last:int
    begin:float
    end:float
    center:float
    groups:int=1
    depth:int=0
    grid:tuple=(9,12)

    def copy(self):return replace(self,features=self.features.clone(),mass=self.mass.clone(),coords=self.coords.clone())

    def metadata(self):
        return dict(first=self.first,last=self.last,begin=self.begin,end=self.end,center=self.center,
            groups=self.groups,depth=self.depth,tokens=len(self.features),original_pairs=self.last-self.first+1,
            mass=float(self.mass.sum()),surviving_source_pair_ids=sorted(set(self.coords[:,0].tolist())),
            tensor_bytes=sum(t.numel()*t.element_size() for t in (self.features,self.mass,self.coords)),
            exact_pair=self.first==self.last and len(self.features)==math.prod(self.grid) and bool(torch.all(self.mass==1)),
            grid=list(self.grid),span_is_provenance_not_full_detail=True)


def validate(n):
    if not isinstance(n,Node) or n.grid not in ((6,8),(9,12),(12,16)):raise ValueError('registered original grid required')
    x,m,c=n.features,n.mass,n.coords
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.requires_grad for t in (x,m,c)):
        raise ValueError('owned CPU inference tensors required')
    if (x.dtype!=torch.float32 or x.ndim!=2 or not 1<=len(x)<=math.prod(n.grid) or not 1<=x.shape[1]<=2048
        or m.dtype!=torch.float32 or m.shape!=(len(x),1) or c.dtype!=torch.long or c.shape!=(len(x),3)
        or not torch.isfinite(x).all() or not torch.isfinite(m).all() or bool(torch.any(m<=0))
        or len(torch.unique(c,dim=0))!=len(c)):
        raise ValueError('finite bounded features, masses and unique coordinates required')
    if (type(n.first) is not int or type(n.last) is not int or not 0<=n.first<=n.last<=2**31-1
        or bool(torch.any(c[:,0]<n.first)) or bool(torch.any(c[:,0]>n.last)) or bool(torch.any(c[:,1:]<0))
        or bool(torch.any(c[:,1]>=n.grid[0])) or bool(torch.any(c[:,2]>=n.grid[1]))
        or not 1<=n.groups<=n.last-n.first+1 or n.depth<0
        or any(not math.isfinite(t) or t<0 for t in (n.begin,n.end,n.center)) or not n.begin<=n.center<=n.end):
        raise ValueError('ordered source coordinates and provenance required')
    keys=c[:,0]*math.prod(n.grid)+c[:,1]*n.grid[1]+c[:,2]
    if bool(torch.any(keys[1:]<=keys[:-1])):raise ValueError('original causal/grid order required')


def pair_node(features,index,begin,end,grid,indices=None):
    count=math.prod(grid)
    if features.ndim!=2 or len(features)!=count:raise ValueError('complete projected pair required')
    if indices is None:indices=torch.arange(count)
    coords=torch.tensor([(index,y,x) for y in range(grid[0]) for x in range(grid[1])],dtype=torch.long)
    n=Node(features.detach().to(device='cpu',dtype=torch.float32)[indices].clone(),
        torch.ones((len(indices),1)),coords[indices].clone(),index,index,float(begin),float(end),(begin+end)/2,grid=grid)
    validate(n);return n


def metric(x,heads=4):
    if x.shape[1]%heads:raise ValueError('fixed metric head alignment required')
    return F.normalize(x.reshape(len(x),heads,-1).mean(1),dim=-1,eps=1e-12)


@torch.no_grad()
def merge(left,right,within=False):
    validate(left);validate(right)
    if left.grid!=right.grid or left.last+1!=right.first or left.end>=right.begin or left.features.shape[1]!=right.features.shape[1]:
        raise ValueError('adjacent nodes with the same grid/width required')
    x,mass,coords=[torch.cat((getattr(left,k),getattr(right,k))) for k in ('features','mass','coords')]
    values=metric(x);scores=values[::2]@values[1::2].T
    best,destination=scores.max(1);remove=len(x)//2
    order=torch.argsort(best,descending=True,stable=True);chosen,unmerged=order[:remove],order[remove:]
    weighted=(x[1::2]*mass[1::2]).scatter_add(0,destination[chosen,None].expand(-1,x.shape[1]),x[::2][chosen]*mass[::2][chosen])
    weights=mass[1::2].scatter_add(0,destination[chosen,None],mass[::2][chosen])
    output=torch.cat((x[::2][unmerged],weighted/weights));output_mass=torch.cat((mass[::2][unmerged],weights))
    output_coords=torch.cat((coords[::2][unmerged],coords[1::2]))
    h,w=left.grid
    permutation=torch.argsort(output_coords[:,0]*(h*w)+output_coords[:,1]*w+output_coords[:,2],stable=True)
    na,nb=left.last-left.first+1,right.last-right.first+1
    parent=Node(output[permutation],output_mass[permutation],output_coords[permutation],left.first,right.last,
        left.begin,right.end,(left.center*na+right.center*nb)/(na+nb),
        max(left.groups,right.groups) if within else left.groups+right.groups,max(left.depth,right.depth)+1,left.grid)
    validate(parent);return parent


def penalties(nodes,now):
    rows=[]
    for a,b in zip(nodes,nodes[1:]):
        value=metric(torch.cat((a.features,b.features)));scores=value[::2]@value[1::2].T;n=len(value)//2
        similarity=float(scores.max(1).values.topk(n).values.mean())
        rows.append((1-similarity,(a.groups+b.groups)/2,(a.center+b.center)/2/max(now,1e-12)))
    maximum=max(r[1] for r in rows)
    return [dict(similarity=s,merge_count=c/maximum+1e-6,time=t,total=.4*s+.4*(c/maximum+1e-6)+.2*t) for s,c,t in rows]


def digest(nodes):
    h=hashlib.sha256()
    for n in nodes:
        h.update(json.dumps(n.metadata(),sort_keys=True).encode())
        for t in (n.features,n.mass,n.coords):h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()
