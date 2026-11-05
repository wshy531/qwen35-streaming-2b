"""Short actual-file 1x checks; fake and random tiny observers, no real weights."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from clock_input import ROOT,LONG,PacketReader,playback_module,clock_run,PLAYBACK

OUT=ROOT/'runs/2b_memory_readability_prepare_20261020_01'
PREV=ROOT/'runs/2b_long_stream_prepare_20261020_01'


@dataclass
class Packet:
    frame_id:int
    timestamp:float
    rgb:object
    decoded_count:int


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def expected_digest(frames):
    h=hashlib.sha256()
    for f in frames:
        item=dict(frame_id=f['frame_id'],timestamp=f['pts_s'],decoded_count=f['frame_id']+1,
            shape=[192,256,3],rgb_sha256=f['rgb_sha256'])
        h.update((json.dumps(item,sort_keys=True,separators=(',',':'))+'\n').encode())
    return dict(samples=len(frames),sha256=h.hexdigest())


class FakeObserver:
    def __init__(self,delay):self.pairs=0;self.last_pts=None;self.delay=delay
    def status(self):return dict(pairs=self.pairs,last_pts=self.last_pts)
    def append_pair(self,a,b):
        assert self.last_pts is None or self.last_pts<a.timestamp<b.timestamp
        self.pairs+=1;self.last_pts=b.timestamp;return self.status()
    def ask(self,q,*,read_from,on_token):
        before=self.status();time.sleep(self.delay)
        on_token(dict(token_index=0,token_id=-1,cumulative_text='CPU_AUDIT',monotonic_s=time.monotonic()))
        assert self.status()==before
        return dict(text='CPU_AUDIT',generated_ids=[-1])


def check(name,spec,observer,samples,times,route):
    module=playback_module();reader=PacketReader(ROOT/spec['media_path'],spec,Packet)
    published=[];trace=[];emitted=[];peak=0
    def publish(packet,state):
        assert state['wall_seconds']+1e-6>=packet.timestamp
        published.append(dict(frame_id=packet.frame_id,pts_s=packet.timestamp,**state))
    def event(e):
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss);assert peak<1536*2**20
        trace.append(e)
    play=module.ClockPlayback(reader,float(samples),samples,on_frame=publish)
    qs=[dict(id=f'{name}:{i}',query_time_s=t,question='CPU interface question, not scored.',
        options=['First option','Second option']) for i,t in enumerate(times)]
    result=clock_run(play,qs,observer,event,emitted.append,read_from=route)
    assert result['source']['source_fingerprint']==expected_digest(spec['frames'][:samples])
    assert result['source']['produced_samples']==samples and result['source']['cancelled_samples']==0
    assert result['source']['delivered_samples']==result['received']
    assert len([e for e in trace if e['kind']=='query'])==len(qs)
    assert all(e['committed_through_s']<=e['requested_s'] for e in trace if e['kind']=='query')
    assert result['wall_seconds']<30
    result.update(peak_RSS_observed=peak,actual_published=published,events=trace,token_events=emitted,
        pretrained_model_loads=0,GPU_jobs=0)
    save(name+'.json',result);print(dict(condition=name,received=result['received'],
        drops=result['source']['dropped_samples'],questions=result['questions'],seconds=result['wall_seconds']),flush=True)
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    start=time.monotonic();here=Path(__file__).parent
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(here),'-p','test_clock_input.py','-v'],
        capture_output=True,text=True,timeout=20)
    save('CPU_clock_contracts.json',dict(exit_code=tests.returncode,stdout=tests.stdout,stderr=tests.stderr))
    assert tests.returncode==0
    spec=json.loads((PREV/'stream_sources6.json').read_text(encoding='utf-8'))[0]
    normal=check('actual12_normal',spec,FakeObserver(0.),12,[2.2,5.5,8.5,10.5],'audit')
    pressure=check('actual12_QA_pressure',spec,FakeObserver(2.4),12,[2.2,5.5,8.5,10.5],'audit')
    assert pressure['source']['dropped_samples']>0
    assert pressure['source']['source_fingerprint']==normal['source']['source_fingerprint']
    sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
    import runtime_bridge
    runtime_bridge.REPO=ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.ACTIVE=runtime_bridge.REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    env=runtime_bridge.runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    from observer import Observer
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](build.processor))).float().eval()
    obs=Observer(env,model,build,policy='M2',main_tokens=384,question_tokens=128,answer_tokens=4)
    tiny=check('actual6_random_tiny',spec,obs,6,[2.2,4.5,5.5],'bank');obs.close()
    assert not torch.cuda.is_initialized()
    save('CPU_clock_receipt.json',dict(status='passed',actual_source=spec['source_path'],
        conditions=3,short_contracts=4,actual_sample_publications=30,
        fake_answers=8,tiny_answers=3,random_tiny_instances=1,pretrained_model_loads=0,GPU_jobs=0,
        no_future_PTS_in_queries=True,producer_continues_during_QA=True,
        source_fingerprint_unchanged_under_QA_pressure=True,
        source_threads_joined=True,seconds=time.monotonic()-start,
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [PLAYBACK,LONG/'replay.py',here/'clock_input.py',here/'cpu_clock.py',here/'test_clock_input.py']},
        limitations=['12sec/6sec engineering checks, not eight-minute pretrained inference',
          'normal/pressure uses fake observer, random tiny does not measure capability',
          'wall-clock runs can drop different frames, not common-input accuracy comparison']))
    print(dict(status='passed',seconds=time.monotonic()-start,pretrained_loads=0,GPU_jobs=0),flush=True)


if __name__=='__main__':main()
