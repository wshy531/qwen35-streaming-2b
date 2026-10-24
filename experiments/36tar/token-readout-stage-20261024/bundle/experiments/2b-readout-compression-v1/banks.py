"""Three unchanged384-token banks, each with full or ephemeral half readout."""
from dataclasses import replace
import hashlib,time
import torch
from common import POLICIES,ARMS
from policy_adapters import BoundedForest,BoundedShared
from features import raw_block
from online_bank import nodes_digest,load_distance,THRESHOLD
from forest import validate

def half_nodes(nodes):
    result=[]
    for node in nodes:
        validate(node)
        if len(node.features)%2 or len(node.features)<2:raise ValueError('registered even node required')
        ids=torch.arange(1,len(node.features),2)
        chosen=replace(node,features=node.features[ids].clone(),mass=node.mass[ids].clone(),coords=node.coords[ids].clone())
        validate(chosen);result.append(chosen)
    return result

class Banks:
    def __init__(self):
        self.banks={p:BoundedShared(48) if p=='LS' else BoundedForest(p=='A06',48) for p in POLICIES}
        self.previous=self.descriptor=None
        self.distance=load_distance();self.pairs=0;self.last_pts=None;self.invalid=False
        self.update_seconds={p:0. for p in POLICIES};self.selection_seconds=0.

    @torch.inference_mode()
    def append(self,features,index,begin,end):
        if self.invalid or index!=self.pairs or (self.last_pts is not None and begin<=self.last_pts):
            raise ValueError('new consecutive causal pair required')
        block=raw_block(features,index,begin,end);desc=block.features.mean(0)
        distance=None if self.descriptor is None else self.distance(self.descriptor,desc)
        starts=self.descriptor is None or distance>THRESHOLD
        try:
            for p,bank in self.banks.items():
                start=time.monotonic();bank.append(block,starts_event=starts)
                self.update_seconds[p]+=time.monotonic()-start
            self.descriptor=desc.clone();self.pairs+=1;self.last_pts=end
            assert all(b.status()['tokens']<=384 for b in self.banks.values())
        except BaseException:self.invalid=True;raise
        return dict(pair_index=index,begin_s=begin,end_s=end,cold_start=index==0,starts_event=starts,
            adjacent_distance=distance,selected_indices=None,temporal_scores=None)

    def export(self,arm):
        if self.invalid or arm not in ARMS:raise ValueError('registered active readout required')
        policy,mode=arm.split('_');nodes=self.banks[policy].export()
        return nodes if mode=='full48' else half_nodes(nodes)
    def fingerprint(self):return {p:nodes_digest(b.export()) for p,b in self.banks.items()}
    def selector_digest(self):
        return hashlib.sha256(b'' if self.descriptor is None else self.descriptor.numpy().tobytes()).hexdigest()
    def status(self):
        return dict(pairs=self.pairs,last_pts=self.last_pts,invalid=self.invalid,
            banks={p:b.status() for p,b in self.banks.items()},
            descriptor_bytes=0 if self.descriptor is None else self.descriptor.numel()*self.descriptor.element_size(),
            no_full_history_feature_list=True,no_main_language_cache=True,readout_is_ephemeral=True)
    def clear(self):self.banks.clear();self.previous=self.descriptor=None;self.invalid=True
