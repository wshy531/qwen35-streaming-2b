"""Five bounded recent windows; preserve selected values, order and source xy."""
from collections import deque
from dataclasses import replace
import hashlib
import time
import torch
from common import ARMS
from features import raw_block
from policy_adapters import pair_node
from selection import select
from online_bank import nodes_digest


class Window:
    def __init__(self, pairs):
        self.recent = deque(maxlen=pairs)

    def append(self, block):
        self.recent.append(pair_node(block))

    def export(self):
        return [n.copy() for n in self.recent]

    def status(self):
        return dict(tokens=sum(len(n.features) for n in self.recent),
                    resident_tensor_bytes=sum(n.metadata()['tensor_bytes'] for n in self.recent),
                    nodes=[n.metadata() for n in self.recent])


class Banks:
    def __init__(self):
        self.banks = {a: Window(int(a.split('_')[1])) for a in ARMS}
        self.previous = None
        self.pairs = 0
        self.last_pts = None
        self.invalid = False
        self.update_seconds = {a: 0. for a in ARMS}
        self.selection_seconds = 0.

    @torch.inference_mode()
    def append(self, features, index, begin, end):
        if self.invalid or index != self.pairs or (self.last_pts is not None and begin <= self.last_pts):
            raise ValueError('active session and consecutive new pair required')
        block = raw_block(features, index, begin, end)
        started = time.monotonic()
        selected, scores = select(block.features, self.previous)
        self.selection_seconds += time.monotonic()-started
        blocks = dict(full48=block)
        for mode in ('odd24', 'change24'):
            ids = selected[mode]
            blocks[mode] = replace(block, features=block.features[ids].clone(),
                                  mass=block.mass[ids].clone(), xy=block.xy[ids].clone())
        cold = self.previous is None
        try:
            for arm, bank in self.banks.items():
                started = time.monotonic()
                bank.append(blocks[arm.split('_')[0]])
                self.update_seconds[arm] += time.monotonic()-started
            self.previous = block.features.clone()
            self.pairs += 1
            self.last_pts = end
            assert all(s['visual_tokens'] <= 384 for s in self.status()['banks'].values())
            assert all(self.status()['banks'][a]['visual_tokens'] <= 192 for a in ('odd24_8','change24_8'))
        except BaseException:
            self.invalid = True
            raise
        return dict(pair_index=index, begin_s=begin, end_s=end, cold_start=cold,
                    selected_indices={m:selected[m].tolist() for m in ('odd24','change24')},
                    temporal_scores=None if scores is None else scores.tolist(),
                    # Compatibility fields only; no event grouping is used.
                    starts_event=cold, adjacent_distance=None, **self.status())

    def export(self, arm):
        if self.invalid:
            raise RuntimeError('invalid bank')
        return self.banks[arm].export()

    def selector_digest(self):
        return hashlib.sha256(b'' if self.previous is None else self.previous.numpy().tobytes()).hexdigest()

    def fingerprint(self):
        return {a:nodes_digest(self.export(a)) for a in ARMS}

    def status(self):
        previous_bytes = 0 if self.previous is None else self.previous.numel()*self.previous.element_size()
        result = {}
        for arm, bank in self.banks.items():
            state = bank.status()
            extra = previous_bytes if arm.startswith('change24') else 0
            result[arm] = dict(visual_tokens=state['tokens'], details=state,
                              resident_tensor_bytes=state['resident_tensor_bytes'],
                              standalone_selector_descriptor_bytes=extra,
                              standalone_persistent_tensor_bytes=state['resident_tensor_bytes']+extra)
        return dict(pairs=self.pairs, last_pts=self.last_pts, invalid=self.invalid, banks=result,
                    shared_selector_bytes=previous_bytes, shared_descriptor_bytes=0,
                    no_full_history_feature_list=True, no_main_language_cache=True)

    def clear(self):
        self.banks.clear()
        self.previous = None
        self.invalid = True
