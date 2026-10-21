"""Actual clock labels retain recording gaps; no pixels or event invention."""
from fractions import Fraction
import math,re


def clock_seconds(value,fps):
    if not isinstance(value,str) or not re.fullmatch(r'\d{8}',value):
        raise ValueError('HHMMSSFF string required')
    rate=Fraction(fps)
    if rate<=0:raise ValueError('actual frame rate required')
    h,m,s,f=(int(value[:2]),int(value[2:4]),int(value[4:6]),int(value[6:]))
    if h>23 or m>59 or s>59 or f>=rate:raise ValueError('invalid source clock')
    return Fraction(h*3600+m*60+s)+Fraction(f,1)/rate


class PairGate:
    """Model consumer gate; producer must have already selected1FPS real samples.

    A gap discards an unmatched sample; it never makes a fabricated pair. A
    question may read only committed pairs, even if a later packet was decoded.
    """
    def __init__(self,max_pair_gap_s=1.5):
        self.max_gap=max_pair_gap_s;self.pending=None;self.last_seen=None
        self.last_committed=None;self.pairs=0;self.discarded_pending=0

    def arrive(self,frame_id,pts):
        if type(frame_id) is not int or frame_id<0 or not math.isfinite(pts) or pts<0:
            raise ValueError('real finite sample identity required')
        if self.last_seen is not None and (frame_id<=self.last_seen[0] or pts<=self.last_seen[1]):
            raise ValueError('sample order/identity must increase')
        self.last_seen=(frame_id,pts)
        if self.pending is not None and pts-self.pending[1]>self.max_gap:
            self.pending=None;self.discarded_pending+=1
        if self.pending is None:self.pending=(frame_id,pts);return None
        first=self.pending;self.pending=None;self.last_committed=pts;self.pairs+=1
        return first,(frame_id,pts)

    def query(self,cutoff):
        if not math.isfinite(cutoff) or cutoff<0:raise ValueError('explicit query time required')
        if self.last_committed is not None and self.last_committed>cutoff:
            raise ValueError('cannot query earlier than already observed state')
        return dict(committed_pairs=self.pairs,committed_through_s=self.last_committed,
            pending_is_not_model_evidence=self.pending is not None)

    def end(self):
        if self.pending is not None:self.discarded_pending+=1;self.pending=None
