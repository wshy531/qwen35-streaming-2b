"""Short real wall-clock scheduling checks, not pretrained model capability."""
from dataclasses import dataclass
import json
import time
import unittest

import numpy as np

from clock_input import clock_run,playback_module


@dataclass
class Packet:
    frame_id:int
    timestamp:float
    rgb:object
    decoded_count:int


class Reader:
    def __init__(self,n=12,step=.04,error=False):self.n=n;self.step=step;self.closed=False;self.error=error
    def __iter__(self):
        for i in range(self.n):
            if self.error and i==3:raise RuntimeError('injected read error')
            yield Packet(i,i*self.step,np.zeros((8,8,3),dtype=np.uint8),i+1)
    def close(self):self.closed=True


class Observer:
    def __init__(self,delay=0.):self.pairs=0;self.last_pts=None;self.delay=delay;self.queries=[]
    def status(self):return dict(pairs=self.pairs,last_pts=self.last_pts)
    def append_pair(self,a,b):
        assert self.last_pts is None or self.last_pts<a.timestamp<b.timestamp
        self.last_pts=b.timestamp;self.pairs+=1;return self.status()
    def ask(self,question,*,read_from,on_token):
        before=self.status();time.sleep(self.delay);self.queries.append(before)
        on_token(dict(token_index=0,token_id=1,cumulative_text='AUDIT',monotonic_s=time.monotonic()))
        assert self.status()==before
        return dict(text='AUDIT',generated_ids=[1])


def questions(times):
    return [dict(id=str(i),query_time_s=t,question='What happened?',options=['A','B']) for i,t in enumerate(times)]


class Tests(unittest.TestCase):
    def test_deadlines_with_overload_and_multiple_due_questions(self):
        mod=playback_module();reader=Reader(n=20);play=mod.ClockPlayback(reader,.8,20)
        obs=Observer(.16);trace=[];tokens=[]
        result=clock_run(play,questions([.09,.15,.15,.4,.6]),obs,trace.append,tokens.append,read_from='audit')
        qs=[e for e in trace if e['kind']=='query']
        self.assertEqual([e['requested_s'] for e in qs],[.09,.15,.15,.4,.6])
        self.assertTrue(all(e['committed_through_s'] is None or e['committed_through_s']<=e['requested_s'] for e in qs))
        self.assertGreater(result['source']['dropped_samples'],0)
        self.assertEqual(result['questions'],5);self.assertTrue(reader.closed)
        self.assertTrue(any(e['dispatch_lag_s']>.05 for e in qs))
        self.assertEqual(qs[1]['observation_state'],qs[2]['observation_state'])

    def test_query_before_pair_and_after_producer_eof(self):
        mod=playback_module();reader=Reader(n=2,step=.03);play=mod.ClockPlayback(reader,.3,2)
        trace=[];obs=Observer()
        result=clock_run(play,questions([0.,.2]),obs,trace.append,lambda e:None,read_from='audit')
        qs=[e for e in trace if e['kind']=='query']
        self.assertEqual(qs[0]['answer']['status'],'no_committed_pair')
        self.assertEqual(qs[1]['committed_through_s'],.03)
        self.assertGreaterEqual(result['wall_seconds'],.2)

    def test_source_error_propagates_and_closes(self):
        mod=playback_module();reader=Reader(error=True);play=mod.ClockPlayback(reader,1.,12)
        with self.assertRaisesRegex(RuntimeError,'injected read error'):
            clock_run(play,questions([.7]),Observer(),lambda e:None,lambda e:None,read_from='audit')
        self.assertTrue(reader.closed);self.assertFalse(play.status()['producer_alive'])

    def test_consumer_failure_stops_producer(self):
        mod=playback_module();reader=Reader(n=100);play=mod.ClockPlayback(reader,4.,100)
        class Broken(Observer):
            def append_pair(self,a,b):raise ValueError('injected model failure')
        with self.assertRaisesRegex(ValueError,'model failure'):
            clock_run(play,questions([2.]),Broken(),lambda e:None,lambda e:None,read_from='audit')
        self.assertTrue(reader.closed);self.assertFalse(play.status()['producer_alive'])


if __name__=='__main__':unittest.main()
