"""Causal PTS sampling with one pending real RGB frame; no video/frame archive."""
from dataclasses import dataclass
import hashlib
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
    decoded_count:int

    def record(self):
        return dict(frame_id=self.frame_id,pts_s=self.timestamp,rgb_sha256=hashlib.sha256(self.rgb.tobytes()).hexdigest())


def fit_canvas(rgb,canvas):
    height,width=canvas;image=Image.fromarray(rgb)
    scale=min(width/image.width,height/image.height)
    size=(max(1,min(width,round(image.width*scale))),max(1,min(height,round(image.height*scale))))
    resized=image.resize(size,Image.Resampling.BICUBIC);result=Image.new('RGB',(width,height),(0,0,0))
    result.paste(resized,((width-size[0])//2,(height-size[1])//2))
    return np.asarray(result)


class CausalSource:
    def __init__(self,path,*,config='auto4',canvas=(288,384),fps=.5):
        if config not in ('auto4','slice1') or canvas not in ((192,256),(288,384)) or fps!=.5:
            raise ValueError('fixed decoder/canvas/sampling protocol required')
        self.path,self.config,self.canvas=path,config,canvas;self.interval=1/fps
        self.container=av.open(str(path));self.stream=self.container.streams.video[0]
        self.stream.codec_context.thread_count=4 if config=='auto4' else 1
        self.stream.codec_context.thread_type='AUTO' if config=='auto4' else 'SLICE'
        self.iterator=iter(self.container.decode(self.stream));self.lookahead=None
        self.origin=self.previous=None;self.due=0.;self.last_query=-1.
        self.decoded=self.sampled=0;self.eof=self.closed=False
        self.decode_seconds=self.transform_seconds=0.

    def _next(self):
        began=time.monotonic()
        try:frame=next(self.iterator)
        except StopIteration:
            self.decode_seconds+=time.monotonic()-began;self.eof=True;return None
        self.decode_seconds+=time.monotonic()-began
        if frame.time is None:raise ValueError('actual display PTS required')
        if self.origin is None:self.origin=float(frame.time)
        pts=float(frame.time)-self.origin
        if not math.isfinite(pts) or pts<0 or (self.previous is not None and pts<=self.previous):
            raise ValueError('strict actual display PTS required')
        self.previous=pts;index=self.decoded;self.decoded+=1
        if self.decoded>20000:raise RuntimeError('frozen subset decode bound exceeded')
        return index,pts,frame

    def advance(self,query_time):
        if self.closed or not math.isfinite(query_time) or query_time<self.last_query or query_time<0:
            raise ValueError('active source and nondecreasing query time required')
        self.last_query=query_time
        while not self.eof:
            item=self.lookahead if self.lookahead is not None else self._next();self.lookahead=None
            if item is None:break
            index,pts,frame=item
            if pts>query_time+1e-9:
                self.lookahead=item;break
            if pts+1e-9<self.due:continue
            began=time.monotonic()
            rgb=fit_canvas(frame.to_ndarray(format='rgb24'),self.canvas)
            self.transform_seconds+=time.monotonic()-began;self.sampled+=1
            if self.sampled>1024:raise RuntimeError('frozen sample metadata limit exceeded')
            self.due=(math.floor((pts+1e-9)/self.interval)+1)*self.interval
            yield Packet(index,pts,rgb,index+1)

    def status(self):
        return dict(decoded_frames=self.decoded,sampled_frames=self.sampled,origin_pts=self.origin,
            eof=self.eof,closed=self.closed,decode_seconds=self.decode_seconds,transform_seconds=self.transform_seconds,
            decoder_config=self.config,decoder_can_read_ahead=True,lookahead_AV_frames=int(self.lookahead is not None),
            future_RGB_emitted=False)

    def close(self):
        self.lookahead=None;self.iterator=None;self.container.close();self.closed=True


class PairDriver:
    def __init__(self,source,on_pair):
        self.source,self.on_pair=source,on_pair;self.pending=None
        self.pairs=self.real_frames=0;self.last_committed=None;self.tail_padded=False
        self.frame_hash=hashlib.sha256();self.closed=False

    def advance(self,query_time):
        if self.closed:raise RuntimeError('closed driver')
        for packet in self.source.advance(query_time):
            assert packet.timestamp<=query_time+1e-9
            self.frame_hash.update(str(packet.record()).encode());self.real_frames+=1
            if self.pending is None:self.pending=packet
            else:
                self.on_pair(self.pending,packet,padding=False)
                self.pending=None;self.pairs+=1;self.last_committed=packet.timestamp
        if self.source.eof and self.pending is not None:
            self.on_pair(self.pending,self.pending,padding=True)
            self.last_committed=self.pending.timestamp;self.pending=None;self.pairs+=1;self.tail_padded=True
        return dict(query_time=query_time,real_frames=self.real_frames,pairs=self.pairs,
            pending_real_frames=int(self.pending is not None),last_committed_pts=self.last_committed,
            submitted_prefix_sha256=self.frame_hash.hexdigest(),true_EOF_tail_padded=self.tail_padded,
            prefix_hash_covers_received_samples_including_pending=True,
            maximum_pending_RGB_bytes=math.prod(self.source.canvas)*3,
            lag_seconds=None if self.last_committed is None else query_time-self.last_committed)

    def close(self):self.pending=None;self.source.close();self.closed=True
