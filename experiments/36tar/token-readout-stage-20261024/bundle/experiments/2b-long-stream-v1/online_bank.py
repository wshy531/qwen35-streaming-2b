"""One persistent bounded bank, updated once per arriving real frame pair.

Reuses frozen M0/M2 algorithms; does not collect full-history feature lists.
Readout returns owned copies and must not mutate the observation bank.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

BASE=Path(__file__).resolve().parents[1]/'2b-natural-eval-v1'
sys.path.insert(0,str(BASE))
from features import raw_block
from forest import raw
from memory384 import Forest384
from memory_policy import Pair,RecentFIFO

THRESHOLD=0.047536224126815796


def load_distance():
    path=BASE/'diag/event_memory_v1/memory.py'
    spec=importlib.util.spec_from_file_location('long_stream_frozen_distance',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.distance


def nodes_digest(nodes):
    h=hashlib.sha256()
    for node in nodes:
        h.update(json.dumps(node.metadata(),sort_keys=True).encode())
        for t in (node.features,node.mass,node.coords):h.update(t.numpy().tobytes())
    return h.hexdigest()


class OnlineBank:
    def __init__(self,policy):
        if policy not in ('M0','M2'):raise ValueError('fixed M0 or M2 required')
        self.policy=policy
        self.bank=RecentFIFO(8) if policy=='M0' else Forest384()
        self.distance=load_distance() if policy=='M2' else None
        self.previous=None;self.last_index=-1;self.last_end=None;self.width=None

    @torch.inference_mode()
    def append(self,features,index,begin,end):
        if (index!=self.last_index+1 or (self.last_end is not None and begin<=self.last_end)
                or (self.width is not None and features.shape[1]!=self.width)):
            raise ValueError('one new consecutive nonoverlapping pair required')
        block=raw_block(features,index,begin,end)
        if self.policy=='M0':
            self.bank.append(Pair(index,begin,end,raw(block)))
            changed=None;starts=None
        else:
            desc=block.features.mean(0)
            changed=None if self.previous is None else self.distance(self.previous,desc)
            starts=self.previous is None or changed>THRESHOLD
            self.bank.append(block,starts_event=starts)
            self.previous=desc.clone()
        self.last_index=index;self.last_end=end;self.width=block.features.shape[1]
        s=self.status()
        assert s['visual_tokens']<=384
        assert s['feature_tensor_bytes']<=3156480
        return dict(pair_index=index,begin_s=begin,end_s=end,adjacent_distance=changed,
            starts_event=starts,**s)

    def export(self):
        if self.policy=='M0':return [p.payload.copy() for p in self.bank.read()]
        return self.bank.export()

    def status(self):
        if self.policy=='M0':
            nodes=[p.payload for p in self.bank.read()]
            meta=[n.metadata() for n in nodes]
            tokens=sum(m['tokens'] for m in meta)
            size=sum(m['tensor_bytes'] for m in meta)
            details=dict(nodes=meta)
        else:
            details=self.bank.status();tokens=details['tokens'];size=details['resident_tensor_bytes']
        return dict(policy=self.policy,processed_pairs=self.last_index+1,
            visual_tokens=tokens,feature_tensor_bytes=size,
            descriptor_bytes=0 if self.previous is None else self.previous.numel()*self.previous.element_size(),
            details=details,no_full_history_feature_list=True)

    def fingerprint(self):return nodes_digest(self.export())
