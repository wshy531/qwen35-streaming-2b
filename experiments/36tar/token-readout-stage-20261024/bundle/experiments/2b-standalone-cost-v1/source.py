"""Exact short-video sampling with stage costs, no retained RGB history."""
import hashlib,time
from common import sha


class ExactPackets:
    def __init__(self,env,path,frames,media_sha):
        self.env,self.path,self.frames,self.media_sha=env,path,frames,media_sha
        self.decode_seconds=self.transform_seconds=self.hash_seconds=self.open_seconds=0.
        self.decoded_frames=self.sampled=0

    def packets(self):
        began=time.monotonic();assert sha(self.path)==self.media_sha
        self.hash_seconds+=time.monotonic()-began
        wanted={f['frame_id']:f for f in self.frames};last=max(wanted);origin=None
        start=time.monotonic();container=self.env['av'].open(str(self.path))
        self.open_seconds+=time.monotonic()-start
        try:
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            iterator=iter(container.decode(stream))
            while True:
                began=time.monotonic()
                try:frame=next(iterator)
                except StopIteration:
                    self.decode_seconds+=time.monotonic()-began;break
                self.decode_seconds+=time.monotonic()-began
                index=self.decoded_frames;self.decoded_frames+=1
                assert frame.time is not None
                if origin is None:origin=float(frame.time)
                if index in wanted:
                    began=time.monotonic();spec=wanted[index];pts=float(frame.time)-origin
                    rgb=self.env['canvas_bytes'](frame)
                    assert abs(pts-spec['pts_s'])<1e-7 and hashlib.sha256(rgb).hexdigest()==spec['rgb_sha256']
                    packet=self.env['FramePacket'](index,pts,self.env['np'].frombuffer(rgb,dtype='uint8').reshape(192,256,3).copy(),index+1)
                    self.sampled+=1;self.transform_seconds+=time.monotonic()-began
                    yield packet
                if index>=last:break
        finally:
            container.close()
        assert self.sampled==len(self.frames)

    def status(self):
        return dict(decode_seconds=self.decode_seconds,transform_seconds=self.transform_seconds,
                    media_integrity_sha_seconds=self.hash_seconds,open_seconds=self.open_seconds,
                    decoded_frames=self.decoded_frames,sampled=self.sampled)
