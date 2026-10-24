"""Bounded actual-PTS RGB producer and causal replay driver.

This is deterministic causal replay, not wall-clock playback. The scheduler can
decode one lookahead frame; it is not passed to the observer before its event.
Only native full-prefix references may retain old RGB, in their own process.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import av
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]/'2b-natural-eval-v1'))
from audit_sampling import canvas_bytes
from score_memory_eval import prompt
from schedule import Frame,Query,events


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


class VideoSource:
    def __init__(self,path,spec,packet_type,*,max_seconds=600):
        self.path=Path(path);self.spec=spec;self.packet_type=packet_type
        self.max_seconds=max_seconds;self.buffer={};self.peak_buffers=0
        self.metadata={};self.sampled=0;self.decoded_frames=0;self.completed=False
        self.decode_seconds=0.;self.transform_seconds=0.;self.hash_seconds=0.

    def frames(self):
        start=time.monotonic();assert sha(self.path)==self.spec['media_sha256']
        self.hash_seconds+=time.monotonic()-start
        previous=origin=None;due=0.;compute_seconds=0.
        with av.open(str(self.path)) as c:
            stream=c.streams.video[0];stream.codec_context.thread_count=1
            self.metadata=dict(fps=float(stream.average_rate),total_num_frames=stream.frames)
            decoder=iter(c.decode(stream))
            while True:
                began=time.monotonic()
                try:frame=next(decoder)
                except StopIteration:break
                elapsed=time.monotonic()-began;self.decode_seconds+=elapsed;compute_seconds+=elapsed
                assert compute_seconds<self.max_seconds,'decode compute budget exceeded'
                i=self.decoded_frames;self.decoded_frames+=1
                if frame.time is None:raise ValueError('missing actual timestamp')
                if origin is None:origin=float(frame.time)
                pts=float(frame.time)-origin
                if previous is not None and pts<=previous:raise ValueError('non-increasing PTS')
                previous=pts
                if pts+1e-9<due:continue
                began=time.monotonic();rgb=canvas_bytes(frame)
                meta=dict(frame_id=i,pts_s=pts,rgb_sha256=hashlib.sha256(rgb).hexdigest())
                assert self.sampled<len(self.spec['frames']) and meta==self.spec['frames'][self.sampled],meta
                assert i not in self.buffer
                self.buffer[i]=self.packet_type(i,pts,np.frombuffer(rgb,dtype=np.uint8).reshape(192,256,3).copy(),i+1)
                self.peak_buffers=max(self.peak_buffers,len(self.buffer))
                assert self.peak_buffers<=3,'unbounded RGB source buffer'
                self.sampled+=1;due=math.floor(pts+1e-9)+1.
                elapsed=time.monotonic()-began;self.transform_seconds+=elapsed;compute_seconds+=elapsed
                yield Frame(i,pts)
        assert self.sampled==len(self.spec['frames'])
        began=time.monotonic();assert sha(self.path)==self.spec['media_sha256']
        self.hash_seconds+=time.monotonic()-began;self.completed=True

    def take_pair(self,ids):return tuple(self.buffer.pop(i) for i in ids)


def replay(source,rows,observer,query_snapshots,on_event,on_token,*,read_from):
    """Observe each real pair once, query in place, then continue same observer."""
    ordered=sorted(rows,key=lambda q:(q['query_time_s'],q['id']))
    by_id={q['id']:q for q in ordered}
    assert len(by_id)==len(ordered)
    expected={q['query_id']:q for q in query_snapshots}
    prefix=hashlib.sha256(b'[');count=0;answers=0;source_name=source.spec['source_path']
    for e in events(source.frames(),[Query(q['id'],q['query_time_s']) for q in ordered],end_s=source.spec['duration_s']):
        if e['kind']=='pair':
            packets=source.take_pair(e['frame_ids'])
            state=observer.append_pair(*packets)
            del packets
            for m in source.spec['frames'][count:count+2]:
                if count:prefix.update(b', ')
                prefix.update(json.dumps(m,sort_keys=True).encode());count+=1
            on_event(dict(source_path=source_name,event=e,state=state))
        elif e['kind']=='query':
            identity=prefix.copy();identity.update(b']')
            snap=expected[e['query_id']]
            assert e=={k:snap[k] for k in e},'query schedule drift'
            assert identity.hexdigest()==snap['committed_frames_sha256'],'committed prefix mismatch'
            assert observer.status()['last_pts']==e['committed_through_s']<=e['query_time_s']
            row=by_id[e['query_id']]
            text=prompt(row)
            result=observer.ask(text,read_from=read_from,
                on_token=lambda item:on_token(dict(source_path=source_name,query_id=row['id'],**item)))
            assert observer.status()['pairs']==e['committed_pairs']
            on_event(dict(source_path=source_name,event=e,committed_frames_sha256=identity.hexdigest(),
                observation_state=observer.status(),answer=result))
            answers+=1
        elif e['kind']=='end':
            assert source.completed and count==e['committed_pairs']*2
            assert set(source.buffer)==({e['pending_frame_id']} if e['pending_frame_id'] is not None else set())
            source.buffer.clear()
            on_event(dict(source_path=source_name,event=e,state=observer.status(),
                producer=dict(peak_RGB_buffers=source.peak_buffers,sampled=source.sampled,
                    decode_seconds=source.decode_seconds,transform_seconds=source.transform_seconds,
                    hash_seconds=source.hash_seconds)))
    assert answers==len(rows)
    return dict(answers=answers,pairs=count//2,source_path=source_name,peak_RGB_buffers=source.peak_buffers)
