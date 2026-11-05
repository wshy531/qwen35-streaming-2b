"""Six bounded banks share one causal descriptor. No full-history feature list."""
import time
import torch

from common import ARMS
from shared import SharedHistory384
from features import raw_block
from forest import raw
from memory384 import Forest384
from policy import DepthForest384
from policies import ReservoirRecent
from online_bank import OnlineBank, load_distance, THRESHOLD, nodes_digest


class Banks:
    def __init__(self):
        self.banks = dict(M0=OnlineBank('M0'), reservoir4=ReservoirRecent(), M2=Forest384(),
            A06=DepthForest384(), LS=SharedHistory384(), LS_depth=SharedHistory384(depth_first=True))
        self.distance = load_distance()
        self.previous = None
        self.pairs = 0
        self.last_pts = None
        self.update_seconds = {a: 0. for a in ARMS}

    @torch.inference_mode()
    def append(self, features, index, begin, end):
        if index != self.pairs or (self.last_pts is not None and begin <= self.last_pts):
            raise ValueError('new causal complete pair required')
        block = raw_block(features, index, begin, end)
        desc = block.features.mean(0)
        changed = None if self.previous is None else self.distance(self.previous, desc)
        starts = self.previous is None or changed > THRESHOLD
        for arm in ARMS:
            started = time.monotonic()
            bank = self.banks[arm]
            if arm == 'M0':
                bank.append(features, index, begin, end)
            elif arm == 'reservoir4':
                bank.append(raw(block))
            else:
                bank.append(block, starts_event=starts)
            self.update_seconds[arm] += time.monotonic() - started
        self.previous = desc.clone()
        self.pairs += 1
        self.last_pts = end
        state = self.status()
        assert all(s['visual_tokens'] <= 384 for s in state['banks'].values())
        return dict(starts_event=starts, adjacent_distance=changed, **state)

    def export(self, arm):
        return self.banks[arm].export()

    def fingerprint(self):
        return {arm: nodes_digest(self.export(arm)) for arm in ARMS}

    def status(self):
        result = {}
        for arm, bank in self.banks.items():
            detail = bank.status()
            if arm == 'M0':
                tokens, size = detail['visual_tokens'], detail['feature_tensor_bytes']
            elif arm == 'reservoir4':
                tokens, size = detail['tokens'], detail['tensor_bytes']
            else:
                tokens, size = detail['tokens'], detail['resident_tensor_bytes']
            result[arm] = dict(visual_tokens=tokens, resident_tensor_bytes=size, details=detail)
        return dict(pairs=self.pairs, last_pts=self.last_pts, banks=result,
            shared_descriptor_bytes=0 if self.previous is None else self.previous.numel()*self.previous.element_size(),
            descriptor_accounting='one shared causal event descriptor; charge it to M2/A06/LS/LS_depth in standalone estimates',
            no_main_language_cache=True, no_full_history_feature_list=True)

    def clear(self):
        self.banks.clear()
        self.previous = None
