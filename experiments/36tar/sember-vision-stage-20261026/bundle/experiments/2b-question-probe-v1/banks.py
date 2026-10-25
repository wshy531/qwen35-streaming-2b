"""One unchanged LS shared384 bank; selection only happens after a question."""
import hashlib
import time
from dataclasses import replace
import torch
from common import ARMS
from policy_adapters import BoundedShared
from features import raw_block
from online_bank import nodes_digest, load_distance, THRESHOLD


class Banks:
    def __init__(self):
        self.banks = {'LS': BoundedShared(48)}
        self.previous = self.descriptor = None
        self.distance = load_distance()
        self.pairs = 0
        self.last_pts = None
        self.invalid = False
        self.update_seconds = {'LS': 0.}
        self.selection_seconds = 0.

    @torch.inference_mode()
    def append(self, features, index, begin, end):
        if self.invalid or index != self.pairs or (self.last_pts is not None and begin <= self.last_pts):
            raise ValueError('new consecutive causal pair required')
        block = raw_block(features, index, begin, end)
        desc = block.features.mean(0)
        distance = None if self.descriptor is None else self.distance(self.descriptor, desc)
        starts = self.descriptor is None or distance > THRESHOLD
        try:
            started = time.monotonic()
            self.banks['LS'].append(block, starts_event=starts)
            self.update_seconds['LS'] += time.monotonic()-started
            self.descriptor = desc.clone()
            self.pairs += 1
            self.last_pts = end
            assert self.banks['LS'].status()['tokens'] <= 384
        except BaseException:
            self.invalid = True
            raise
        return dict(pair_index=index, begin_s=begin, end_s=end, cold_start=index == 0,
                    starts_event=starts, adjacent_distance=distance, selected_indices=None, temporal_scores=None)

    def full(self):
        if self.invalid:
            raise RuntimeError('closed bank')
        return self.banks['LS'].export()

    def export(self, arm):
        if arm not in ARMS or arm == 'LS_probe24':
            raise ValueError('query-aware selection requires explicit scores')
        nodes = self.full()
        if arm == 'LS_full48':
            return nodes
        return [replace(n, features=n.features[1::2].clone(), mass=n.mass[1::2].clone(),
                        coords=n.coords[1::2].clone()) for n in nodes]

    def fingerprint(self):
        return {'LS': nodes_digest(self.full())}

    def selector_digest(self):
        return hashlib.sha256(b'' if self.descriptor is None else self.descriptor.numpy().tobytes()).hexdigest()

    def status(self):
        return dict(pairs=self.pairs, last_pts=self.last_pts, invalid=self.invalid,
                    banks={p: b.status() for p, b in self.banks.items()},
                    descriptor_bytes=0 if self.descriptor is None else self.descriptor.numel()*self.descriptor.element_size(),
                    no_full_history_feature_list=True, no_main_language_cache=True, readout_is_ephemeral=True)

    def clear(self):
        self.banks.clear()
        self.previous = self.descriptor = None
        self.invalid = True
