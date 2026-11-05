"""Same forest formulas, per-view budgets and one fixed 192-to-108 selector."""
import hashlib
import math
import torch
import torch.nn.functional as F
from resolution_common import SPECS,THRESHOLD,arms_for
from resolution_nodes import pair_node,digest
from resolution_banks_base import Bank,distance


def select108(features,previous):
    if features.shape[0]!=192 or features.device.type!='cpu':raise ValueError('complete high grid required')
    uniform=torch.div(torch.arange(108)*192,108,rounding_mode='floor')
    if previous is None:return uniform,uniform.clone(),True
    if features.shape!=previous.shape:raise ValueError('matching high reference required')
    scores=(1-(F.normalize(features,dim=-1,eps=1e-12)*F.normalize(previous,dim=-1,eps=1e-12)).sum(-1)).clamp(0,2)
    scores[(features.norm(dim=-1)<=1e-12)&(previous.norm(dim=-1)<=1e-12)]=0
    return uniform,torch.argsort(scores,descending=True,stable=True)[:108].sort().values,False


class Banks:
    def __init__(self,grid=(9,12),*,arms=None):
        self.grid=tuple(grid);canvas=tuple(n*32 for n in grid)
        if canvas not in ((288,384),(384,512)):raise ValueError('two registered resolutions only')
        arms=arms_for(canvas) if arms is None else tuple(arms)
        if not arms or len(set(arms))!=len(arms) or any(a not in arms_for(canvas) for a in arms):raise ValueError('registered explicit arms required')
        self.banks={a:Bank(SPECS[a]['policy'],SPECS[a]['keep'],budget=SPECS[a]['budget'],
            recent=SPECS[a]['recent'],closed=SPECS[a]['closed']) for a in arms}
        self.needs_previous=any('change' in a for a in arms)
        self.previous=self.descriptor=None;self.pairs=0;self.last_end=None;self.invalid=False

    @torch.inference_mode()
    def append(self,features,index,begin,end):
        if self.invalid or index!=self.pairs or (self.last_end is not None and begin<=self.last_end):raise ValueError('ordered pair required')
        features=features.detach().to(device='cpu',dtype=torch.float32)
        if features.ndim!=2 or len(features)!=math.prod(self.grid) or not torch.isfinite(features).all():raise ValueError('complete finite projection required')
        desc=features.mean(0);changed=None if self.descriptor is None else distance(self.descriptor,desc)
        starts=self.descriptor is None or changed>THRESHOLD
        indices={'mid108':torch.arange(108)} if self.grid==(9,12) else {'high192':torch.arange(192)}
        cold=False
        if self.grid==(12,16):
            uniform,change,cold=select108(features,self.previous)
            indices.update(highuniform108=uniform,highchange108=change)
        nodes={view:pair_node(features,index,begin,end,self.grid,ids) for view,ids in indices.items()
            if any(SPECS[a]['view']==view for a in self.banks)}
        try:
            for arm,bank in self.banks.items():bank.append(nodes[SPECS[arm]['view']],starts_event=starts)
        except BaseException:self.invalid=True;raise
        self.previous=features.clone() if self.needs_previous else None
        self.descriptor=desc.clone();self.pairs+=1;self.last_end=end
        return dict(pair_index=index,begin_s=begin,end_s=end,starts_event=starts,adjacent_distance=changed,
            cold_start=cold,selected_indices={v:i.tolist() for v,i in indices.items() if '108' in v and v!='mid108'})

    def export(self,arm):
        if self.invalid:raise RuntimeError('closed banks')
        return self.banks[arm].export()

    def fingerprint(self):return {a:digest(b.nodes()) for a,b in self.banks.items()}

    def selector_digest(self):
        h=hashlib.sha256()
        for t in (self.previous,self.descriptor):
            if t is not None:h.update(t.numpy().tobytes())
        return h.hexdigest()

    def status(self):
        prev=0 if self.previous is None else self.previous.numel()*self.previous.element_size()
        desc=0 if self.descriptor is None else self.descriptor.numel()*self.descriptor.element_size()
        result={}
        for arm,bank in self.banks.items():
            state=bank.status();extra=(prev if 'change' in arm else 0)+(desc if bank.policy!='M0' else 0)
            result[arm]=dict(visual_tokens=state['tokens'],standalone_extra_state_bytes=extra,
                standalone_persistent_tensor_bytes=state['resident_tensor_bytes']+extra,details=state)
        return dict(pairs=self.pairs,last_end=self.last_end,grid=list(self.grid),banks=result,
            shared_previous_bytes=prev,shared_descriptor_bytes=desc,no_full_history_feature_list=True,no_main_language_cache=True)

    def clear(self):self.banks.clear();self.previous=self.descriptor=None;self.invalid=True
