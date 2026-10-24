"""Causal sampled-frame/query ordering, without model or media dependencies.

Only committed pairs are visible to QA. An odd pending frame is reported;
it is never completed using a post-query frame or silently padded at query.
This is a scheduler contract, not a model integration or real-time benchmark.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Frame:
    frame_id: int
    pts: float


@dataclass(frozen=True)
class Query:
    query_id: str
    pts: float


def finite_time(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0


def events(frames,queries,*,end_s):
    """Consume frames lazily, with at most one lookahead held by the caller.

    Query ordering is fixed before inference. The backend must receive only
    receive/pair events, and must isolate QA state. No backend is called here.
    """
    if not finite_time(end_s):raise ValueError('finite stream end required')
    qs=tuple(queries)
    if any(not isinstance(q,Query) or not q.query_id or not finite_time(q.pts) or q.pts>end_s for q in qs):
        raise ValueError('queries must be within the stream')
    if len({q.query_id for q in qs})!=len(qs):raise ValueError('unique query IDs required')
    if any(a.pts>b.pts for a,b in zip(qs,qs[1:])):raise ValueError('ordered query times required')
    cursor=0;last=None;pending=None;committed=None;pairs=0;received=0

    def question(q):
        return dict(kind='query',query_id=q.query_id,query_time_s=q.pts,
            received_frames=received,committed_pairs=pairs,
            committed_through_s=None if committed is None else committed.pts,
            pending_frame_id=None if pending is None else pending.frame_id,
            observation_age_s=None if committed is None else q.pts-committed.pts,
            pair_policy='complete_pair_only_no_query_padding',
            no_visible_pair=committed is None)

    for frame in frames:
        if not isinstance(frame,Frame) or type(frame.frame_id) is not int or frame.frame_id<0 or not finite_time(frame.pts):
            raise ValueError('valid frame metadata required')
        if frame.pts>end_s:raise ValueError('sample exceeds stream end')
        if last is not None and (frame.frame_id<=last.frame_id or frame.pts<=last.pts):
            raise ValueError('strictly increasing frame IDs and timestamps required')
        while cursor<len(qs) and qs[cursor].pts<frame.pts:
            yield question(qs[cursor]);cursor+=1
        yield dict(kind='receive',frame_id=frame.frame_id,pts_s=frame.pts)
        received+=1;last=frame
        if pending is None:pending=frame
        else:
            yield dict(kind='pair',pair_index=pairs,frame_ids=[pending.frame_id,frame.frame_id],
                begin_s=pending.pts,end_s=frame.pts)
            pairs+=1;committed=frame;pending=None
    while cursor<len(qs):
        yield question(qs[cursor]);cursor+=1
    yield dict(kind='end',end_s=end_s,received_frames=received,committed_pairs=pairs,
        pending_frame_id=None if pending is None else pending.frame_id,
        incomplete_tail_committed=False)
