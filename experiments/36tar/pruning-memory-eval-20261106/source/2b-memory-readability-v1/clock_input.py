"""Deadline-priority QA over the existing PTS-paced single-slot producer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
LONG=ROOT/'experiments/2b-long-stream-v1'
sys.path.insert(0,str(LONG))
from replay import VideoSource
from score_memory_eval import prompt

PLAYBACK=ROOT/'publish/qwen35-streaming-2b/experiments/36tar/streaming-learning/code/qwen35-streaming-s4/streaming/playback.py'


def playback_module(path=PLAYBACK):
    spec=importlib.util.spec_from_file_location('owned_pts_playback',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


class PacketReader:
    def __init__(self,path,spec,packet_type):
        self.source=VideoSource(path,spec,packet_type)
        self.iterator=iter(self.source.frames())
    def __iter__(self):return self
    def __next__(self):
        frame=next(self.iterator)
        return self.source.buffer.pop(frame.frame_id)
    @property
    def metadata(self):return self.source.metadata
    def close(self):self.iterator.close();self.source.buffer.clear()


def clock_run(playback,rows,observer,on_event,on_token,*,read_from,clock=time.monotonic):
    """Model has one owner. Due questions precede any newer-frame append.

    In-flight observation may finish after a question's wall deadline; only
    source PTS <= its timestamp can be read. Dispatch/answer lag is explicit.
    The producer continues during QA. This is not a common-input accuracy run.
    """
    qs=sorted(rows,key=lambda q:(q['query_time_s'],q['id']))
    if len({q['id'] for q in qs})!=len(qs):raise ValueError('unique questions required')
    if any(not 0<=q['query_time_s']<playback.duration for q in qs):raise ValueError('question outside clock window')
    cursor=0;pending=None;received=committed=0;last_pts=last_id=None
    prefix=hashlib.sha256(b'[');peak_age=0.;max_gap=0.
    started=False
    try:
        playback.start();started=True
        while True:
            now=clock()-playback.origin
            if cursor<len(qs) and now>=qs[cursor]['query_time_s']:
                row=qs[cursor];qtime=row['query_time_s'];dispatch=now;first=last=None
                assert last_pts is None or last_pts<=qtime,'future committed evidence'
                def emit(item):
                    nonlocal first,last
                    wall=clock()-playback.origin
                    if first is None:first=wall
                    last=wall;on_token(dict(query_id=row['id'],requested_s=qtime,
                        emitted_wall_s=wall,**item))
                identity=prefix.copy();identity.update(b']')
                if last_pts is None:
                    answer=dict(status='no_committed_pair',text='',generated_ids=[])
                else:
                    answer=observer.ask(prompt(row),read_from=read_from,on_token=emit)
                    assert observer.status()['last_pts']==last_pts
                age=None if last_pts is None else qtime-last_pts
                if age is not None:peak_age=max(peak_age,age)
                on_event(dict(kind='query',query_id=row['id'],requested_s=qtime,dispatch_wall_s=dispatch,
                    dispatch_lag_s=dispatch-qtime,committed_through_s=last_pts,
                    observation_age_at_request_s=age,pending_frame_id=None if pending is None else pending.frame_id,
                    first_token_after_request_s=None if first is None else first-qtime,
                    answer_available_after_request_s=clock()-playback.origin-qtime,
                    committed_frames_sha256=identity.hexdigest(),observation_state=observer.status(),answer=answer))
                cursor+=1;continue
            cutoff=qs[cursor]['query_time_s'] if cursor<len(qs) else float('inf')
            mailbox=playback.mailbox;packet=None
            with mailbox.condition:
                if mailbox.error is not None:raise RuntimeError('producer failed: '+mailbox.error)
                if mailbox.latest is not None and mailbox.latest.timestamp<=cutoff:
                    # Recheck deadline after acquiring the lock; a pending
                    # question always wins over starting another observation.
                    if cursor<len(qs) and clock()-playback.origin>=cutoff:continue
                    packet=mailbox.latest;mailbox.latest=None;mailbox.delivered+=1
                elif mailbox.done and mailbox.latest is None and cursor==len(qs):break
                else:
                    remaining=.05 if cursor==len(qs) else max(0.,min(.05,cutoff-(clock()-playback.origin)))
                    mailbox.condition.wait(timeout=remaining);continue
            assert packet.timestamp<=clock()-playback.origin+1e-6,'producer published future PTS'
            assert last_id is None or packet.frame_id>last_id
            last_id=packet.frame_id;received+=1
            on_event(dict(kind='receive',frame_id=packet.frame_id,pts_s=packet.timestamp,wall_s=clock()-playback.origin))
            if pending is None:pending=packet;continue
            a,b=pending,packet;pending=None
            assert a.timestamp<b.timestamp
            state=observer.append_pair(a,b)
            committed+=2;last_pts=b.timestamp;max_gap=max(max_gap,b.timestamp-a.timestamp)
            for p in (a,b):
                if committed-2 or p is b:prefix.update(b', ')
                prefix.update(json.dumps(dict(frame_id=p.frame_id,pts_s=p.timestamp,
                    rgb_sha256=hashlib.sha256(p.rgb.tobytes()).hexdigest()),sort_keys=True).encode())
            on_event(dict(kind='pair',frame_ids=[a.frame_id,b.frame_id],pts_s=[a.timestamp,b.timestamp],
                completed_wall_s=clock()-playback.origin,state=state))
            del a,b,packet
        assert received==committed+int(pending is not None)
        on_event(dict(kind='end',received=received,committed_pairs=committed//2,
            uncommitted_tail_id=None if pending is None else pending.frame_id,
            maximum_pair_gap_s=max_gap,maximum_query_observation_age_s=peak_age))
    finally:
        if started:playback.close()
    status=playback.status()
    assert not status['producer_alive']
    assert status['produced_samples']==status['delivered_samples']+status['dropped_samples']+status['cancelled_samples']+status['buffered_samples']
    return dict(questions=cursor,received=received,pairs=committed//2,
        maximum_query_observation_age_s=peak_age,source=status,wall_seconds=clock()-playback.origin,
        no_query_padding=True,accuracy_inputs_may_differ_across_wallclock_runs=True)
