"""A bounded, causal visual bank with isolated fresh-cache question readout."""
import hashlib
import json

from .frozen import THRESHOLD, policies

FOREST_POLICIES = ('M2', 'M2_A06')
EVENT_POLICIES = (*FOREST_POLICIES, 'LS_shared384_v1')


def digest(nodes):
    # Identical serialization to the historical fixed_stage2.node_digest.
    h = hashlib.sha256()
    for node in nodes:
        h.update(json.dumps(node.metadata(), sort_keys=True).encode())
        for value in (node.features, node.mass, node.coords):
            h.update(value.numpy().tobytes())
    return h.hexdigest()


class MemorySession:
    """Owns one policy only; no full-history feature container or language cache.

    readout(nodes, question) must finish and release its own temporary cache.
    A pair is committed once; a duplicated final frame marks a terminal odd tail.
    """
    def __init__(self, policy, readout=None):
        if policy not in ('M0', 'reservoir4', *EVENT_POLICIES):
            raise ValueError('M0, M2, reservoir4, M2_A06 or LS_shared384_v1 required')
        self.policy, self.readout = policy, readout
        self.legacy = policies()
        self.reset()

    def reset(self):
        if self.policy == 'LS_shared384_v1':
            from .shared import SharedHistory384
            self.bank = SharedHistory384()
        elif self.policy == 'M2_A06':
            from .a06 import DepthForest384
            self.bank = DepthForest384()
        else:
            self.bank = (self.legacy.fifo.RecentFIFO(8) if self.policy == 'M0' else
                         self.legacy.forest384.Forest384() if self.policy == 'M2' else
                         self.legacy.reservoir.ReservoirRecent())
        self.previous = None
        self.last_index, self.last_end, self.width = -1, None, None
        self.last_frame_id = -1
        self.closed, self.valid, self.terminal_tail = False, True, False
        self.append_count = self.ask_count = 0

    def _check_open(self):
        if self.closed or not self.valid:
            raise RuntimeError('session closed or invalid; reset required')

    def _nodes(self):
        if self.policy == 'M0':
            return [pair.payload for pair in self.bank.read()]
        if self.policy in EVENT_POLICIES:
            return self.bank.nodes()
        return sorted([*self.bank.old, *self.bank.recent], key=lambda n: n.first)

    def export(self):
        self._check_open()
        return [node.copy() for node in self._nodes()]

    def state_digest(self):
        h = hashlib.sha256(digest(self._nodes()).encode())
        if self.previous is not None:
            h.update(self.previous.numpy().tobytes())
        h.update(str((self.last_index, self.last_end, self.width, self.last_frame_id,
                      self.terminal_tail)).encode())
        h.update(json.dumps(self._controls(),sort_keys=True).encode())
        return h.hexdigest()

    def _controls(self):
        if self.policy == 'LS_shared384_v1':
            return self.bank.controls()
        if self.policy in FOREST_POLICIES:
            controls = dict(last_index=self.bank.last_index,last_end=self.bank.last_end,
                        width=self.bank.width,within_merges=self.bank.within_merges,
                        long_merges=self.bank.long_merges,
                        recent_event_starts=[flag for _,flag in self.bank.recent])
            if self.policy == 'M2_A06':
                controls['selection_count'] = self.bank.selection_count
            return controls
        if self.policy == 'reservoir4':
            return dict(promoted=self.bank.promoted,last=self.bank.last,
                        rng_state=self.bank.rng.getstate())
        last=self.bank.last
        return dict(capacity=self.bank.capacity,last=None if last is None else
                    [last.index,last.start_s,last.end_s])

    def append_pair(self, block, *, frame_ids, padded_tail=False):
        self._check_open()
        if type(padded_tail) is not bool or len(frame_ids) != 2 or any(
                type(i) is not int or i < 0 for i in frame_ids):
            raise ValueError('two real frame IDs and boolean padding flag required')
        node = self.legacy.forest.raw(block)  # validate before touching policy state
        if (self.terminal_tail or node.first != self.last_index + 1
                or (self.last_end is not None and node.begin <= self.last_end)
                or (self.width is not None and node.features.shape[1] != self.width)
                or frame_ids[0] <= self.last_frame_id
                or (padded_tail and (frame_ids[0] != frame_ids[1] or node.begin != node.end))
                or (not padded_tail and (frame_ids[0] >= frame_ids[1] or node.begin >= node.end))):
            raise ValueError('sequential causal pair with fixed width required')
        change = None
        if self.policy in EVENT_POLICIES:
            descriptor = node.features.mean(0)
            change = None if self.previous is None else self.legacy.distance(self.previous, descriptor)
            starts = self.previous is None or change > THRESHOLD
        before = self.status()
        try:
            if self.policy == 'M0':
                self.bank.append(self.legacy.fifo.Pair(node.first, node.begin, node.end, node))
            elif self.policy in EVENT_POLICIES:
                self.bank.append(block, starts_event=starts)
                self.previous = descriptor.clone()
            else:
                self.bank.append(node)
            self.last_index, self.last_end = node.last, node.end
            self.width, self.last_frame_id = node.features.shape[1], frame_ids[1]
            self.terminal_tail = padded_tail
            self.append_count += 1
            after = self.status()
            if after['visual_tokens'] > 384 or after['metadata_bytes'] > 32768:
                raise RuntimeError('resident bank exceeded fixed budget')
        except BaseException:
            self.valid = False
            raise
        return dict(pair=node.first, frame_ids=list(frame_ids), begin=node.begin, end=node.end,
                    padded_tail=padded_tail, adjacent_distance=change,
                    starts_event=starts if self.policy in EVENT_POLICIES else None,
                    merges_before=before['merge_counts'], merges_after=after['merge_counts'],
                    state=after)

    def ask(self, question):
        self._check_open()
        if not isinstance(question, str) or not question.strip() or self.last_index < 0:
            raise ValueError('nonempty question and at least one committed pair required')
        if self.readout is None:
            raise RuntimeError('no readout supplied')
        before = self.state_digest()
        nodes = self.export()
        exported = digest(nodes)
        try:
            result = self.readout(nodes, question)
        finally:
            if self.state_digest() != before:
                self.valid = False
                raise RuntimeError('question mutated persistent memory')
        if digest(nodes) != exported:
            raise RuntimeError('readout modified its exported input')
        self.ask_count += 1
        return dict(result, node_digest=exported, memory_unchanged=True,
                    node_metadata=[n.metadata() for n in nodes])

    def status(self):
        nodes = self._nodes()
        metadata = [n.metadata() for n in nodes]
        detector_bytes = 0 if self.previous is None else self.previous.numel() * self.previous.element_size()
        return dict(policy=self.policy, closed=self.closed, valid=self.valid,
                    pairs_received=self.append_count, last_pair=self.last_index, last_end=self.last_end,
                    visual_tokens=sum(n['tokens'] for n in metadata), nodes=metadata,
                    tensor_bytes=sum(n['tensor_bytes'] for n in metadata),
                    detector_bytes=detector_bytes,
                    resident_bytes=sum(n['tensor_bytes'] for n in metadata) + detector_bytes,
                    metadata_bytes=len(json.dumps(dict(nodes=metadata,controls=self._controls())).encode()),
                    merge_counts=dict(within=getattr(self.bank, 'within_merges', 0),
                                      cross=getattr(self.bank, 'long_merges', 0)),
                    state_digest=self.state_digest(), full_history_retained=False,
                    language_cache_retained=False)

    def close(self):
        self.reset()
        self.closed = True
