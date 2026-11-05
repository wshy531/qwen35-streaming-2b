"""Exactly eight original pairs per arm; pruning cannot silently expand history."""
from collections import deque
import hashlib
import torch
from quality_common import ARMS,KEEP
from resolution_nodes import Node,validate,digest
from sember_banks import select


def selected_node(features,indices,index,begin,end):
    x=features.detach().to(device='cpu',dtype=torch.float32)
    indices=indices.to(device='cpu',dtype=torch.long)
    if len(x)!=len(indices) or x.ndim!=2:raise ValueError('selected values and original group IDs required')
    coords=torch.stack((torch.full_like(indices,index),indices//12,indices%12),1)
    n=Node(x.clone(),torch.ones((len(x),1)),coords,index,index,float(begin),float(end),(begin+end)/2,grid=(9,12))
    validate(n);return n


class Bank:
    def __init__(self,arm):
        self.arm=arm;self.budget=8*KEEP[arm];self.recent=deque(maxlen=8);self.last_index=-1;self.last_end=None
    def append(self,node):
        validate(node)
        if node.first!=node.last or node.first!=self.last_index+1 or len(node.features)!=KEEP[self.arm]:
            raise ValueError('consecutive fixed-size original pair required')
        if self.last_end is not None and node.begin<=self.last_end:raise ValueError('causal pair time required')
        self.recent.append(node.copy());self.last_index=node.last;self.last_end=node.end
        assert sum(len(n.features) for n in self.recent)<=self.budget
    def nodes(self):return list(self.recent)
    def export(self):return [n.copy() for n in self.recent]
    def status(self):
        meta=[n.metadata() for n in self.recent]
        return dict(tokens=sum(n['tokens'] for n in meta),nodes=meta,source_pairs=len(meta),
            resident_tensor_bytes=sum(n['tensor_bytes'] for n in meta),last_index=self.last_index,
            same_eight_pair_coverage=True,no_merging=True)


class Banks:
    def __init__(self,arms=ARMS):
        if not arms or len(set(arms))!=len(arms) or any(a not in ARMS for a in arms):raise ValueError('registered distinct arms required')
        self.banks={a:Bank(a) for a in arms};self.previous=None;self.pairs=0;self.last_end=None;self.invalid=False
    @torch.inference_mode()
    def append(self,full,mid,index,begin,end):
        if self.invalid or index!=self.pairs or (self.last_end is not None and begin<=self.last_end):raise ValueError('ordered live bank required')
        needs_full=any(a!='mid_odd54_8' for a in self.banks)
        if needs_full:
            full=full.detach().to(device='cpu',dtype=torch.float32)
            if len(full)!=108 or not torch.isfinite(full).all():raise ValueError('complete finite original grid required')
        odd=torch.arange(1,108,2);ids={}
        if 'post_change54_8' in self.banks:ids,_=select(full,self.previous)
        selected={}
        try:
            for arm,bank in self.banks.items():
                if arm=='full108_8':idx=torch.arange(108);x=full
                elif arm=='post_odd54_8':idx=odd;x=full[idx]
                elif arm=='post_change54_8':idx=ids['change'];x=full[idx]
                else:
                    x,idx=mid
                    if not torch.equal(idx.cpu(),odd):raise ValueError('mid-pruned group order changed')
                bank.append(selected_node(x,idx,index,begin,end));selected[arm]=idx.tolist()
        except BaseException:self.invalid=True;raise
        # No full previous pair is retained by the other standalone methods.
        if 'post_change54_8' in self.banks:self.previous=full.clone()
        self.pairs+=1;self.last_end=end
        return dict(pair_index=index,selected_indices=selected,same_coverage=True)
    def fingerprint(self):return {a:digest(b.nodes()) for a,b in self.banks.items()}
    def selector_digest(self):
        return hashlib.sha256(b'' if self.previous is None else self.previous.numpy().tobytes()).hexdigest()
    def status(self):
        extra=0 if self.previous is None else self.previous.numel()*self.previous.element_size()
        return dict(pairs=self.pairs,banks={a:dict(visual_tokens=b.status()['tokens'],details=b.status(),
            standalone_extra_state_bytes=extra if a=='post_change54_8' else 0,
            standalone_persistent_tensor_bytes=b.status()['resident_tensor_bytes']+(extra if a=='post_change54_8' else 0))
            for a,b in self.banks.items()},no_full_history_feature_list=True,no_main_language_cache=True,
            shared_previous_bytes=extra)
    def clear(self):self.banks.clear();self.previous=None;self.invalid=True
