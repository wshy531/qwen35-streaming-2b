"""All six full media streams through production scheduler with a fake reader."""
from dataclasses import dataclass
import json
from pathlib import Path
import time

import psutil

from inventory_local import ROOT,OUT
from replay import VideoSource,replay


@dataclass
class Packet:
    frame_id:int
    timestamp:float
    rgb:object
    received_index:int


class AuditObserver:
    def __init__(self):self.pairs=0;self.last_pts=None;self.last_frame_id=None;self.answers=0
    def status(self):return dict(pairs=self.pairs,last_pts=self.last_pts,last_frame_id=self.last_frame_id)
    def append_pair(self,a,b):
        assert a.timestamp<b.timestamp and a.frame_id<b.frame_id
        assert self.last_pts is None or self.last_pts<a.timestamp
        assert a.rgb.shape==b.rgb.shape==(192,256,3)
        self.last_pts=b.timestamp;self.last_frame_id=b.frame_id;self.pairs+=1
        return self.status()
    def ask(self,question,*,read_from,on_token):
        assert read_from=='audit' and question.endswith('Do not explain.')
        self.answers+=1
        on_token(dict(token_index=0,token_id=-1,cumulative_text='CPU_AUDIT_NOT_MODEL',monotonic_s=time.monotonic()))
        return dict(status='CPU_AUDIT_NOT_MODEL',observation_through_s=self.last_pts)


def main():
    def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
    rows=read('stream_requests34.json');snapshots=read('stream_query_snapshots34.json')
    result=[];events=tokens=0;peak_RSS=0;start=time.monotonic()
    def event(e):
        nonlocal events,peak_RSS
        events+=1;peak_RSS=max(peak_RSS,psutil.Process().memory_info().rss)
        assert peak_RSS<1024*2**20
    def token(e):
        nonlocal tokens
        tokens+=1;assert e['token_index']==0
    for spec in read('stream_sources6.json'):
        source=VideoSource(ROOT/spec['media_path'],spec,Packet)
        audit=AuditObserver()
        r=replay(source,[q for q in rows if q['source_path']==spec['source_path']],audit,
            [q for q in snapshots if q['source_path']==spec['source_path']],event,token,read_from='audit')
        result.append(r);print(r,flush=True)
    assert tokens==sum(r['answers'] for r in result)==34
    receipt=dict(status='passed',sources=result,answers=34,events=events,callback_events=tokens,
        peak_process_RSS_bytes=peak_RSS,seconds=time.monotonic()-start,
        full_real_media_decoded=True,all_sampled_RGB_and_committed_prefix_hashes_match=True,
        pretrained_model_loads=0,random_model_loads=0,GPU_jobs=0,
        limitation='fake observer; not pretrained capability or wall-clock playback')
    with (OUT/'CPU_full_media_replay.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(receipt,flush=True)


if __name__=='__main__':main()
