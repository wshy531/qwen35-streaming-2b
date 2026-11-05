"""Nine bounded experimental banks; no full-history features or node children."""
from collections import deque
from dataclasses import replace
import hashlib,time
import torch
from common import ARMS
from policy_adapters import BoundedForest,BoundedShared,pair_node
from features import raw_block
from selection import select
from online_bank import nodes_digest,load_distance,THRESHOLD

class Recent:
    def __init__(self):self.recent=deque()
    def append(self,block,*,starts_event):
        node=pair_node(block)
        self.recent.append(node)
        while sum(len(n.features) for n in self.recent)>384:self.recent.popleft()
    def export(self):return [n.copy() for n in self.recent]
    def status(self):
        return dict(tokens=sum(len(n.features) for n in self.recent),
                    resident_tensor_bytes=sum(n.metadata()['tensor_bytes'] for n in self.recent),
                    nodes=[n.metadata() for n in self.recent])

class Banks:
    def __init__(self):
        self.banks={}
        for arm in ARMS:
            policy,mode=arm.split('_');size=48 if mode=='full48' else 24
            self.banks[arm]=Recent() if policy=='M0' else BoundedShared(size) if policy=='LS' else BoundedForest(policy=='A06',size)
        self.previous=self.descriptor=None
        self.distance=load_distance()
        self.pairs=0;self.last_pts=None;self.invalid=False
        self.update_seconds={a:0. for a in ARMS}
        self.selection_seconds=0.

    @torch.inference_mode()
    def append(self,features,index,begin,end):
        if self.invalid or index!=self.pairs or (self.last_pts is not None and begin<=self.last_pts):
            raise ValueError('active session and new complete pair required')
        block=raw_block(features,index,begin,end)
        desc=block.features.mean(0)
        changed=None if self.descriptor is None else self.distance(self.descriptor,desc)
        starts=self.descriptor is None or changed>THRESHOLD
        begin_selection=time.monotonic()
        selected,scores=select(block.features,self.previous)
        self.selection_seconds+=time.monotonic()-begin_selection
        blocks={'full48':block}
        for mode in ('odd24','change24'):
            ids=selected[mode]
            blocks[mode]=replace(block,features=block.features[ids].clone(),mass=block.mass[ids].clone(),xy=block.xy[ids].clone())
        cold=self.previous is None
        try:
            for arm,bank in self.banks.items():
                started=time.monotonic()
                bank.append(blocks[arm.split('_')[1]],starts_event=starts)
                self.update_seconds[arm]+=time.monotonic()-started
            self.previous=block.features.clone();self.descriptor=desc.clone()
            self.pairs+=1;self.last_pts=end
            state=self.status()
            assert all(x['visual_tokens']<=384 for x in state['banks'].values())
        except BaseException:
            self.invalid=True;raise
        return dict(pair_index=index,begin_s=begin,end_s=end,cold_start=cold,
            selected_indices={m:selected[m].tolist() for m in ('odd24','change24')},
            temporal_scores=None if scores is None else scores.tolist(),
            starts_event=starts,adjacent_distance=changed,**state)

    def export(self,arm):
        if self.invalid:raise RuntimeError('failed/closed banks')
        return self.banks[arm].export()
    def selector_digest(self):
        h=hashlib.sha256()
        for x in (self.previous,self.descriptor):
            if x is not None:h.update(x.numpy().tobytes())
        return h.hexdigest()
    def fingerprint(self):return {a:nodes_digest(self.export(a)) for a in ARMS}
    def status(self):
        result={}
        prev=0 if self.previous is None else self.previous.numel()*self.previous.element_size()
        desc=0 if self.descriptor is None else self.descriptor.numel()*self.descriptor.element_size()
        for arm,bank in self.banks.items():
            s=bank.status();policy,mode=arm.split('_')
            extra=(prev if mode=='change24' else 0)+(desc if policy!='M0' else 0)
            result[arm]=dict(visual_tokens=s['tokens'],resident_tensor_bytes=s['resident_tensor_bytes'],
                standalone_selector_descriptor_bytes=extra,
                standalone_persistent_tensor_bytes=s['resident_tensor_bytes']+extra,details=s)
        return dict(pairs=self.pairs,last_pts=self.last_pts,invalid=self.invalid,banks=result,
            shared_selector_bytes=prev,shared_descriptor_bytes=desc,
            no_full_history_feature_list=True,no_main_language_cache=True)
    def clear(self):
        self.banks.clear();self.previous=self.descriptor=None;self.invalid=True
