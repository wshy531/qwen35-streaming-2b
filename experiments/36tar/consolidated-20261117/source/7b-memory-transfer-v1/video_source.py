# Derived from streamingbench-interface-v1/source.py; only canvas and AUTO4 decoder settings adapted.
"""Causal 1-FPS PTS sampling with one lookahead frame and one pending RGB sample."""
from dataclasses import dataclass
import math
import time
import av
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Packet:
    frame_id:int
    timestamp:float
    rgb:object


def canvas(rgb):
    # Preserve the current experiment's exact two-step resize protocol.
    image=Image.fromarray(rgb);image.thumbnail((392,392),Image.Resampling.BICUBIC)
    scale=min(392/image.width,280/image.height)
    image=image.resize((max(1,min(392,round(image.width*scale))),max(1,min(280,round(image.height*scale)))),Image.Resampling.BICUBIC)
    result=Image.new('RGB',(392,280));result.paste(image,((392-image.width)//2,(280-image.height)//2))
    return np.asarray(result).copy()


class CausalPairs:
    def __init__(self,path,on_pair):
        if not callable(on_pair):raise ValueError('pair callback required')
        self.container=av.open(str(path));self.stream=self.container.streams.video[0]
        self.stream.codec_context.thread_count=4
        self.stream.codec_context.thread_type='AUTO'
        self.iterator=iter(self.container.decode(self.stream));self.on_pair=on_pair
        self.origin=None;self.previous=-1.;self.due=0.;self.last_query=-1.
        self.lookahead=self.pending=None;self.eof=self.closed=False
        self.decoded=self.real_samples=self.pairs=self.synthetic_samples=0
        self.last_committed=None

    def _next(self):
        try:frame=next(self.iterator)
        except StopIteration:self.eof=True;return None
        if frame.time is None:raise ValueError('actual display PTS required')
        if self.origin is None:self.origin=float(frame.time)
        pts=float(frame.time)-self.origin
        if not math.isfinite(pts) or pts<0 or pts<=self.previous:raise ValueError('strict display PTS order required')
        self.previous=pts;index=self.decoded;self.decoded+=1
        if self.decoded>100000:raise RuntimeError('finite per-video decode limit')
        return index,pts,frame

    def advance(self,query):
        if self.closed or not math.isfinite(query) or query<0 or query<self.last_query:
            raise ValueError('active source and nondecreasing finite query required')
        self.last_query=query
        while not self.eof:
            item=self.lookahead if self.lookahead is not None else self._next();self.lookahead=None
            if item is None:break
            index,pts,frame=item
            if pts>query+1e-9:
                self.lookahead=item;break
            if pts+1e-9<self.due:continue
            sample=Packet(index,pts,canvas(frame.to_ndarray(format='rgb24')))
            self.real_samples+=1;self.due=math.floor(pts+1e-9)+1
            if self.real_samples>4096:raise RuntimeError('finite per-video sampled-frame limit')
            if self.pending is None:self.pending=sample
            else:
                self.on_pair(self.pending,sample,padding=False);self.pending=None
                self.pairs+=1;self.last_committed=pts
        if self.eof and self.pending is not None:
            # Only true EOF permits the old protocol's explicit last-sample duplicate.
            self.on_pair(self.pending,self.pending,padding=True)
            self.last_committed=self.pending.timestamp;self.pending=None
            self.pairs+=1;self.synthetic_samples+=1
        return self.status()

    def status(self):
        return dict(real_samples=self.real_samples,complete_pairs=self.pairs,synthetic_samples=self.synthetic_samples,
            last_committed_pts=self.last_committed,actual_EOF=self.eof,pending_real_samples=int(self.pending is not None),
            lookahead_AV_frames=int(self.lookahead is not None),future_RGB_emitted=False,
            maximum_pending_RGB_bytes=280*392*3,closed=self.closed)

    def close(self):
        if not self.closed:
            self.pending=self.lookahead=None;self.iterator=None;self.container.close();self.closed=True
