"""Same eight uncompressed pairs for each visual-encoder arm; no forest."""
from collections import deque
import time
from common import ARMS
from features import raw_block
from policy_adapters import pair_node
from online_bank import nodes_digest


class Banks:
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('unknown fixed visual arm')
        self.arm = arm
        self.banks = {arm:deque(maxlen=8)}
        self.pairs = 0
        self.last_pts = None
        self.invalid = False
        self.previous = None

    def append(self, features, index, begin, end):
        if self.invalid or index != self.pairs or (self.last_pts is not None and begin <= self.last_pts):
            raise ValueError('new consecutive pair required')
        node = pair_node(raw_block(features, index, begin, end))
        self.banks[self.arm].append(node)
        self.pairs += 1
        self.last_pts = end
        assert self.status()['banks'][self.arm]['visual_tokens'] <= 384
        return dict(pair_index=index, begin_s=begin, end_s=end)

    def export(self, arm):
        if self.invalid or arm != self.arm:
            raise ValueError('invalid bank')
        return [n.copy() for n in self.banks[arm]]

    def selector_digest(self):
        return 'no_selector'

    def fingerprint(self):
        return {self.arm:nodes_digest(self.export(self.arm))}

    def status(self):
        nodes = list(self.banks[self.arm])
        state = dict(visual_tokens=sum(len(n.features) for n in nodes),
                     resident_tensor_bytes=sum(n.metadata()['tensor_bytes'] for n in nodes),
                     nodes=[n.metadata() for n in nodes])
        return dict(pairs=self.pairs, last_pts=self.last_pts, invalid=self.invalid,
                    banks={self.arm:state}, no_main_language_cache=True,
                    no_full_history_feature_list=True)

    def clear(self):
        self.banks.clear()
        self.invalid = True
