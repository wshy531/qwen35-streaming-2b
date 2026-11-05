"""Actual-frame, answer-blind external-memory scheduling on CPU.

This is not a Qwen KV/conv/recurrent-state implementation or feature encoder.
M0/M1 store lightweight pair identities here; M2 cannot be evaluated without
real features. Queries fork policy state so odd-tail padding is never persisted.
"""
import copy
from dataclasses import asdict
import math

from memory_policy import Pair,RecentFIFO,RecentLongHierarchy


class MemoryQueryBlueprint:
    def __init__(self,frames):
        # Make each queued frame an explicit whitelist copy. Never accept GT,
        # evidence or oracle fields as part of pair payloads.
        self.frames=[dict(frame_id=f['frame_id'],pts_s=f['pts_s'],rgb_sha256=f['rgb_sha256']) for f in frames]
        if not self.frames or any(not math.isfinite(f['pts_s']) or f['pts_s']<0 for f in self.frames) or any(a['frame_id']>=b['frame_id'] or a['pts_s']>=b['pts_s']
            for a,b in zip(self.frames,self.frames[1:])):
            raise ValueError('strictly increasing actual frame trace required')
        self.m0=RecentFIFO(8);self.m1=RecentLongHierarchy(4,4)
        self.cursor=0;self.pair_index=0;self.pending=None;self.last_query=None

    def state(self):
        return dict(cursor=self.cursor,pair_index=self.pair_index,pending=copy.deepcopy(self.pending),
            M0=[asdict(p) for p in self.m0.read()],M1=[asdict(p) for p in self.m1.read()],
            M1_provenance=list(self.m1.provenance()))

    def _advance(self,query):
        if isinstance(query,bool) or not isinstance(query,(int,float)) or not math.isfinite(query) or query<0:
            raise ValueError('finite nonnegative query time required')
        if self.last_query is not None and query<self.last_query:
            raise ValueError('queries must be scheduled in chronological order')
        while self.cursor<len(self.frames) and self.frames[self.cursor]['pts_s']<=query:
            current=self.frames[self.cursor];self.cursor+=1
            if self.pending is None:self.pending=current
            else:
                pair=Pair(self.pair_index,self.pending['pts_s'],current['pts_s'],
                    dict(frames=[self.pending,current],snapshot_padding=False))
                self.m0.append(pair);self.m1.append(pair)
                self.pair_index+=1;self.pending=None
        self.last_query=query

    def query(self,query):
        self._advance(query)
        if not self.cursor:raise ValueError('query before first actual frame')
        before=self.state();m0=copy.deepcopy(self.m0);m1=copy.deepcopy(self.m1)
        if self.pending is not None:
            # The duplicate belongs only to independent readout copies.
            tail=Pair(self.pair_index,self.pending['pts_s'],self.pending['pts_s'],
                dict(frames=[self.pending,self.pending],snapshot_padding=True))
            m0.append(tail);m1.append(tail)
        result=dict(query_time_s=query,seen_real_frames=self.cursor,
            committed_real_pairs=self.pair_index,pending_frame_id=self.pending['frame_id'] if self.pending else None,
            M0=[asdict(p) for p in m0.read()],M1=[asdict(p) for p in m1.read()],
            M1_node_provenance=list(m1.provenance()),
            M2_status='not_constructed_requires_real_visual_features_and_adapter',
            main_state_changed_by_readout=False)
        if self.state()!=before:raise AssertionError('query fork mutated persistent memory')
        for name in ('M0','M1'):
            if len(result[name])>8:raise AssertionError('pair budget exceeded')
            if any(f['pts_s']>query for p in result[name] for f in p['payload']['frames']):
                raise AssertionError('future frame in memory readout')
        return result

