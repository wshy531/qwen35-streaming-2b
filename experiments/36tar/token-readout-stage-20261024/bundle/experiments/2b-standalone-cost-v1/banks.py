"""Exactly one recent window; baseline arms have no change selector state."""
from collections import deque
from dataclasses import replace
import hashlib,time
import torch
from common import ARMS
from features import raw_block
from policy_adapters import pair_node
from selection import select,odd_indices
from online_bank import nodes_digest


class Window:
    def __init__(self):self.recent=deque(maxlen=8)
    def export(self):return [n.copy() for n in self.recent]
    def status(self):
        return dict(tokens=sum(len(n.features) for n in self.recent),
                    resident_tensor_bytes=sum(n.metadata()['tensor_bytes'] for n in self.recent),
                    nodes=[n.metadata() for n in self.recent])


class Banks:
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('one registered arm required')
        self.arm=arm;self.banks={arm:Window()};self.previous=None
        self.pairs=0;self.last_pts=None;self.invalid=False
        self.transfer_validate_seconds=self.selection_seconds=0.
        self.update_seconds={arm:0.}

    @torch.inference_mode()
    def append(self,features,index,begin,end):
        if self.invalid or index!=self.pairs or (self.last_pts is not None and begin<=self.last_pts):
            raise ValueError('new consecutive real pair required')
        started=time.monotonic();block=raw_block(features,index,begin,end)
        self.transfer_validate_seconds+=time.monotonic()-started
        cold=self.previous is None;scores=None;ids=None
        if self.arm=='change24_8':
            started=time.monotonic();chosen,scores=select(block.features,self.previous);ids=chosen['change24']
            self.selection_seconds+=time.monotonic()-started
        elif self.arm=='odd24_8':
            started=time.monotonic();ids=odd_indices()
            self.selection_seconds+=time.monotonic()-started
        started=time.monotonic()
        if ids is not None:
            selected=replace(block,features=block.features[ids].clone(),mass=block.mass[ids].clone(),xy=block.xy[ids].clone())
        else:selected=block
        try:
            self.banks[self.arm].recent.append(pair_node(selected))
            if self.arm=='change24_8':self.previous=block.features.clone()
            self.pairs+=1;self.last_pts=end
            cap=384 if self.arm=='full48_8' else 192
            assert self.status()['banks'][self.arm]['visual_tokens']<=cap
            assert self.arm=='change24_8' or self.previous is None
        except BaseException:
            self.invalid=True;raise
        self.update_seconds[self.arm]+=time.monotonic()-started
        return dict(pair_index=index,begin_s=begin,end_s=end,cold_start=index==0,
                    selected_indices={self.arm:(list(range(48)) if ids is None else ids.tolist())},
                    temporal_scores=None if scores is None else scores.tolist(),
                    starts_event=index==0,adjacent_distance=None)

    def export(self,arm):
        if self.invalid or arm!=self.arm:raise ValueError('only the active valid arm exists')
        return self.banks[arm].export()
    def selector_digest(self):
        return hashlib.sha256(b'' if self.previous is None else self.previous.numpy().tobytes()).hexdigest()
    def fingerprint(self):return {self.arm:nodes_digest(self.export(self.arm))}
    def status(self):
        state=self.banks[self.arm].status()
        extra=0 if self.previous is None else self.previous.numel()*self.previous.element_size()
        return dict(pairs=self.pairs,last_pts=self.last_pts,invalid=self.invalid,
                    banks={self.arm:dict(visual_tokens=state['tokens'],details=state,
                        resident_tensor_bytes=state['resident_tensor_bytes'],
                        standalone_persistent_tensor_bytes=state['resident_tensor_bytes']+extra,
                        standalone_selector_descriptor_bytes=extra)},
                    shared_selector_bytes=extra,shared_descriptor_bytes=0,
                    no_full_history_feature_list=True,no_main_language_cache=True)
    def clear(self):self.banks.clear();self.previous=None;self.invalid=True
