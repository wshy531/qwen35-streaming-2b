"""Lazy validated RGB source, configurable decoder threads, no model dependency."""
from dataclasses import dataclass
import hashlib,time
import av
import numpy as np
from audit_sampling import canvas_bytes
from schedule import Frame
from common import CONFIGS,sha


@dataclass
class Packet:
    frame_id:int
    timestamp:float
    rgb:object
    received_serial:int


class BoundedAVSource:
    def __init__(self,path,spec,packet_type=Packet,*,config='slice1',guard=lambda:None):
        if config not in CONFIGS:raise ValueError('one fixed decoder configuration required')
        self.path,self.spec,self.packet_type,self.config=path,spec,packet_type,config
        self.guard=guard;self.buffer={};self.peak_buffers=0
        self.sampled=self.decoded_frames=0;self.completed=False
        self.decode_seconds=self.transform_seconds=self.hash_seconds=self.open_seconds=0.
        self.first_sample_seconds=None;self.metadata={};self._started=False

    def frames(self):
        if self._started:raise RuntimeError('source is single use')
        self._started=True;self.guard()
        began=time.monotonic();assert sha(self.path)==self.spec['sha256']
        self.hash_seconds+=time.monotonic()-began
        expected=self.spec['frames'];wanted={r['frame_id']:r for r in expected}
        assert len(wanted)==len(expected) and expected
        last_wanted=max(wanted);origin=previous=None;start=time.monotonic()
        began=time.monotonic();container=av.open(str(self.path));self.open_seconds+=time.monotonic()-began
        normal=False
        try:
            stream=container.streams.video[0];context=stream.codec_context
            settings=CONFIGS[self.config];context.thread_count=settings['thread_count'];context.thread_type=settings['thread_type']
            self.metadata=dict(codec=context.name,width=context.width,height=context.height,
                               requested_threads=settings['thread_count'],thread_type=str(context.thread_type))
            iterator=iter(container.decode(stream))
            while True:
                began=time.monotonic()
                try:frame=next(iterator)
                except StopIteration:
                    self.decode_seconds+=time.monotonic()-began;break
                self.decode_seconds+=time.monotonic()-began
                index=self.decoded_frames;self.decoded_frames+=1
                if index%64==0:self.guard()
                if frame.time is None:raise ValueError('missing actual presentation timestamp')
                if origin is None:origin=float(frame.time)
                pts=float(frame.time)-origin
                if previous is not None and pts<=previous:raise ValueError('non-increasing actual PTS')
                previous=pts
                if index in wanted:
                    began=time.monotonic();spec=wanted[index]
                    assert abs(pts-spec['pts_s'])<1e-7
                    rgb=canvas_bytes(frame);assert hashlib.sha256(rgb).hexdigest()==spec['rgb_sha256'],dict(frame_id=index,reason='pixel_identity_mismatch')
                    self.buffer[index]=self.packet_type(index,pts,np.frombuffer(rgb,dtype=np.uint8).reshape(192,256,3).copy(),index+1)
                    self.peak_buffers=max(self.peak_buffers,len(self.buffer));assert self.peak_buffers<=3
                    assert index==expected[self.sampled]['frame_id'];self.sampled+=1
                    if self.first_sample_seconds is None:self.first_sample_seconds=time.monotonic()-start
                    self.transform_seconds+=time.monotonic()-began
                    yield Frame(index,pts)
                if self.spec['cohort']!='stream34' and index>=last_wanted:break
            assert self.sampled==len(expected)
            if self.spec['cohort']=='stream34':
                began=time.monotonic();assert sha(self.path)==self.spec['sha256'];self.hash_seconds+=time.monotonic()-began
            self.completed=True;normal=True
        finally:
            container.close()
            if not normal:self.buffer.clear()

    def take_pair(self,ids):return tuple(self.buffer.pop(i) for i in ids)

    def packets(self):
        for frame in self.frames():yield self.take_pair([frame.frame_id])[0]

    def close(self):self.buffer.clear()

    def status(self):
        return dict(completed=self.completed,sampled=self.sampled,decoded_frames=self.decoded_frames,
                    peak_application_RGB_buffers=self.peak_buffers,decode_seconds=self.decode_seconds,
                    transform_seconds=self.transform_seconds,media_integrity_sha_seconds=self.hash_seconds,
                    open_seconds=self.open_seconds,first_sample_seconds=self.first_sample_seconds,codec=self.metadata)
