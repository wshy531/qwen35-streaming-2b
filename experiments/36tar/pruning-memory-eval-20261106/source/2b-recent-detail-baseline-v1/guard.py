"""Borrowed recent-detail baseline, not a new pruner or StreamForest reproduction.

All arms require FULL 108-token projected pairs. No encoder speedup is claimed.
The protected arm stores only the missing 54 latest features, not a duplicate
108-token pair. History policy and boundaries are shared with the 810 control.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-resolution-cost-v1'))
import torch
from resolution_nodes import Node, pair_node, validate, digest
from resolution_banks_base import Bank, distance
from resolution_common import THRESHOLD

ARMS = ('uniform864', 'uniform810', 'protect864')


class RecentDetail:
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('registered baseline required')
        self.arm = arm
        self.bank = Bank('LS', 54, budget=864 if arm == 'uniform864' else 810,
                         recent=432, closed=324)
        self.extra = None
        self.previous = None
        self.closed = False

    @torch.inference_mode()
    def append(self, features, index, begin, end):
        if self.closed:
            raise RuntimeError('closed baseline')
        # Validate before mutating state; complete encoding is mandatory.
        if features.ndim != 2 or features.shape[0] != 108:
            raise ValueError('full 108 projected tokens required; no reconstruction')
        node = pair_node(features, index, begin, end, (9, 12), torch.arange(1, 108, 2))
        if not torch.isfinite(features).all():
            raise ValueError('finite full features required')
        desc = node.features.mean(0)
        starts = self.previous is None or distance(self.previous, desc) > THRESHOLD
        self.bank.append(node, starts_event=starts)
        self.extra = (features[::2].detach().to(device='cpu', dtype=torch.float32).clone()
                      if self.arm == 'protect864' else None)
        self.previous = desc.clone()
        assert self.status()['stored_feature_tokens'] <= 864
        return starts

    def export(self):
        if self.closed:
            raise RuntimeError('closed baseline')
        nodes = self.bank.export()
        if self.extra is None:
            return nodes
        n = nodes[-1]
        assert n.first == n.last == self.bank.last_index and len(n.features) == 54
        assert torch.all(n.mass == 1)
        ids = torch.arange(108)
        full = torch.empty((108, n.features.shape[1]), dtype=torch.float32)
        full[1::2] = n.features
        full[::2] = self.extra
        coords = torch.stack((torch.full_like(ids, n.first), ids // 12, ids % 12), 1)
        nodes[-1] = Node(full, torch.ones((108, 1)), coords, n.first, n.last,
                         n.begin, n.end, n.center, grid=(9, 12))
        validate(nodes[-1])
        return nodes

    def fingerprint(self):
        h = hashlib.sha256(digest(self.bank.nodes()).encode())
        for t in (self.extra, self.previous):
            if t is not None:
                h.update(t.contiguous().numpy().tobytes())
        return h.hexdigest()

    def status(self):
        s = self.bank.status()
        extra = 0 if self.extra is None else self.extra.numel() * self.extra.element_size()
        desc = 0 if self.previous is None else self.previous.numel() * self.previous.element_size()
        tokens = s['tokens'] + (54 if self.extra is not None else 0)
        return dict(arm=self.arm, stored_feature_tokens=tokens, read_visual_tokens=tokens,
                    total_visual_budget=864 if self.arm != 'uniform810' else 810,
                    base_bank=s, latest_complement_bytes=extra, descriptor_bytes=desc,
                    resident_tensor_bytes=s['resident_tensor_bytes'] + extra + desc,
                    full_encoder_required=True, no_language_main_KV=True,
                    no_full_history_archive=True, export_and_encoder_transients_not_in_resident=True)

    def close(self):
        self.bank.recent.clear()
        self.bank.history.clear()
        self.bank.roots.clear()
        self.bank.open = None
        self.extra = self.previous = None
        self.closed = True
