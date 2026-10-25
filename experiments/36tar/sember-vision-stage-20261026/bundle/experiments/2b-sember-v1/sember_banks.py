"""Eleven fixed, bounded conditions, without a full feature archive or main KV."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import torch
import torch.nn.functional as F
from sember_common import ARMS,THRESHOLD
from sember_nodes import pair_node,merge,penalties,digest,validate


def select(features,previous=None):
    if features.device.type!='cpu' or features.ndim!=2 or len(features) not in (48,108) or not torch.isfinite(features).all():
        raise ValueError('complete finite CPU projected pair required')
    count=len(features);odd=torch.arange(1,count,2)
    if previous is None:return {'full':torch.arange(count),'odd':odd,'change':odd.clone()},None
    if previous.shape!=features.shape or previous.device.type!='cpu' or not torch.isfinite(previous).all():
        raise ValueError('matching previous pair required')
    scores=(1-(F.normalize(features.float(),dim=-1,eps=1e-12)*F.normalize(previous.float(),dim=-1,eps=1e-12)).sum(-1)).clamp(0,2)
    scores[(features.norm(dim=-1)<=1e-12)&(previous.norm(dim=-1)<=1e-12)]=0
    ids=torch.argsort(scores,descending=True,stable=True)[:count//2].sort().values
    return {'full':torch.arange(count),'odd':odd,'change':ids},scores


def distance(a,b):
    if float(a.norm())==0 or float(b.norm())==0:return 1.0
    return max(0.,min(2.,1.-float(F.cosine_similarity(a,b,dim=0,eps=1e-12))))


@dataclass(frozen=True)
class Entry:
    node:object
    first_event:int
    last_event:int


class Bank:
    def __init__(self,policy,pair_tokens,*,budget=864,recent=432,closed=324):
        if policy not in ('M0','M2','A06','LS'):raise ValueError('fixed registered policy required')
        if any(type(n) is not int or n<1 for n in (pair_tokens,budget,recent,closed)):raise ValueError('positive budget required')
        self.policy,self.pair_tokens,self.budget=policy,pair_tokens,budget
        self.recent_cap,self.closed_cap=recent,closed
        self.recent=deque();self.roots=[];self.open=None;self.history=[]
        self.last_index=-1;self.last_end=None;self.width=self.grid=None
        self.current_event=-1;self.within_merges=self.long_merges=0

    def nodes(self):
        if self.policy=='LS':return [e.node for e in (*self.history,*self.recent)]
        return [*self.roots,*([self.open] if self.open is not None else []),*[e.node for e in self.recent]]

    def _choose(self,costs):
        if self.policy=='A06':
            return min(range(len(costs)),key=lambda i:(1+max(self.roots[i].depth,self.roots[i+1].depth),costs[i]['total'],i))
        return min(range(len(costs)),key=lambda i:(costs[i]['total'],i))

    @torch.inference_mode()
    def append(self,node,*,starts_event):
        validate(node)
        if (type(starts_event) is not bool or node.first!=node.last or node.first!=self.last_index+1
            or len(node.features)!=self.pair_tokens or (self.last_end is not None and node.begin<=self.last_end)
            or (self.width is not None and (node.features.shape[1]!=self.width or node.grid!=self.grid))):
            raise ValueError('fixed size/grid, consecutive pair and event flag required')
        event=0 if self.last_index==-1 else self.current_event+int(starts_event)
        if self.policy!='M0' and sum(len(e.node.features) for e in self.recent)>=self.recent_cap:
            old=self.recent.popleft()
            if self.policy=='LS':
                self.history.append(old)
                while sum(len(e.node.features) for e in self.history)>self.budget-self.recent_cap:
                    costs=penalties([e.node for e in self.history],old.node.end);i=self._choose(costs)
                    a,b=self.history[i:i+2];within=a.first_event==a.last_event==b.first_event==b.last_event
                    self.history[i:i+2]=[Entry(merge(a.node,b.node,within),a.first_event,b.last_event)]
                    self.within_merges+=int(within);self.long_merges+=int(not within)
            else:
                if old.first_event!=old.last_event:raise RuntimeError('recent entry is not an original pair')
                # Event-start belongs to this promoted pair, not the newest arrival.
                if old.first_event!=getattr(self,'open_event',None) and self.open is not None:
                    self.roots.append(self.open);self.open=None
                if self.open is None:self.open=old.node
                else:self.open=merge(self.open,old.node,True);self.within_merges+=1
                self.open_event=old.first_event
                while sum(len(n.features) for n in self.roots)>self.closed_cap:
                    costs=penalties(self.roots,old.node.end);i=self._choose(costs)
                    self.roots[i:i+2]=[merge(self.roots[i],self.roots[i+1])];self.long_merges+=1
        self.recent.append(Entry(node.copy(),event,event))
        if self.policy=='M0':
            while sum(len(e.node.features) for e in self.recent)>self.budget:self.recent.popleft()
        self.current_event=event;self.last_index=node.last;self.last_end=node.end;self.width=node.features.shape[1];self.grid=node.grid
        state=self.status()
        if state['tokens']>self.budget or state['metadata_bytes']>65536:raise RuntimeError('bank budget exceeded')

    def export(self):return [n.copy() for n in self.nodes()]

    def status(self):
        nodes=self.nodes();meta=[n.metadata() for n in nodes]
        return dict(tokens=sum(len(n.features) for n in nodes),recent_tokens=sum(len(e.node.features) for e in self.recent),
            open_tokens=0 if self.open is None else len(self.open.features),closed_tokens=sum(len(n.features) for n in self.roots),
            history_tokens=sum(len(e.node.features) for e in self.history),nodes=meta,
            resident_tensor_bytes=sum(m['tensor_bytes'] for m in meta),metadata_bytes=len(json.dumps(meta).encode()),
            within_merges=self.within_merges,long_merges=self.long_merges,last_index=self.last_index,
            recent_event_ids=[e.first_event for e in self.recent],
            history_event_ids=[[e.first_event,e.last_event] for e in self.history])


class Banks:
    def __init__(self,grid=(9,12)):
        if grid not in ((6,8),(9,12)):raise ValueError('registered grid required')
        self.grid=grid;full=math.prod(grid);half=full//2;self.banks={}
        for arm in ARMS:
            policy,mode,*coverage=arm.split('_');size=full if mode.startswith('full') else half
            budget=(int(coverage[0])*size if policy=='M0' else full*8)
            self.banks[arm]=Bank(policy,size,budget=budget,recent=full*4,closed=full*3)
        self.previous=self.descriptor=None;self.pairs=0;self.last_end=None;self.invalid=False

    @torch.inference_mode()
    def append(self,features,index,begin,end):
        if self.invalid or index!=self.pairs or (self.last_end is not None and begin<=self.last_end):raise ValueError('new ordered pair required')
        features=features.detach().to(device='cpu',dtype=torch.float32)
        if len(features)!=math.prod(self.grid) or not torch.isfinite(features).all():raise ValueError('complete grid required')
        desc=features.mean(0);changed=None if self.descriptor is None else distance(self.descriptor,desc)
        starts=self.descriptor is None or changed>THRESHOLD
        indices,scores=select(features,self.previous)
        nodes={mode:pair_node(features,index,begin,end,self.grid,ids) for mode,ids in indices.items()}
        try:
            for arm,bank in self.banks.items():
                mode=arm.split('_')[1];bank.append(nodes['full' if mode.startswith('full') else 'odd' if mode.startswith('odd') else 'change'],starts_event=starts)
        except BaseException:self.invalid=True;raise
        self.previous=features.clone();self.descriptor=desc.clone();self.pairs+=1;self.last_end=end
        return dict(pair_index=index,begin_s=begin,end_s=end,starts_event=starts,adjacent_distance=changed,
            selected_indices={mode:ids.tolist() for mode,ids in indices.items() if mode!='full'},
            cold_start=scores is None)

    def export(self,arm):
        if self.invalid:raise RuntimeError('failed or closed bank set')
        return self.banks[arm].export()

    def fingerprint(self):return {a:digest(b.nodes()) for a,b in self.banks.items()}

    def selector_digest(self):
        h=hashlib.sha256()
        for t in (self.previous,self.descriptor):
            if t is not None:h.update(t.numpy().tobytes())
        return h.hexdigest()

    def status(self):
        result={};prev=0 if self.previous is None else self.previous.numel()*self.previous.element_size()
        desc=0 if self.descriptor is None else self.descriptor.numel()*self.descriptor.element_size()
        for arm,bank in self.banks.items():
            state=bank.status();extra=(prev if 'change' in arm else 0)+(desc if bank.policy!='M0' else 0)
            result[arm]=dict(visual_tokens=state['tokens'],standalone_extra_state_bytes=extra,
                standalone_persistent_tensor_bytes=state['resident_tensor_bytes']+extra,details=state)
        return dict(pairs=self.pairs,last_end=self.last_end,grid=list(self.grid),banks=result,
            shared_previous_bytes=prev,shared_descriptor_bytes=desc,no_full_history_feature_list=True,no_main_language_cache=True)

    def clear(self):self.banks.clear();self.previous=self.descriptor=None;self.invalid=True
