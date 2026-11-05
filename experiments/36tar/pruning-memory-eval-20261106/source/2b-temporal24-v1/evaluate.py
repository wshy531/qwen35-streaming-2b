"""Single authorized FP32 job; finite causal archives, no training or media download."""
from collections import Counter
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from common import ROOT, read, save, sha, runtime, ARMS


def verify(run):
    assert run.resolve().is_relative_to((ROOT/'runs').resolve())
    protocol=read(run/'protocol.json')
    for name,digest in protocol['input_sha256'].items():
        assert sha(run/name)==digest,name
    lock=read(run/'execution_lock.json')
    for name,digest in lock['input_sha256'].items():assert sha(run/name)==digest,name
    for name,digest in lock['runtime_sha256'].items():
        p=ROOT/name
        if name.startswith('repo/') and not (ROOT/'repo').exists():p=ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==digest,name
    assert sha(run/'protocol.json')==lock['protocol_sha256']
    rows,groups=read(run/'questions.json'),read(run/'groups.json')
    assert len(rows)==len({q['id'] for q in rows})==303 and len(groups)==112
    return protocol,rows,groups


def packet_stream(env,path,frames):
    wanted={f['frame_id']:f for f in frames};last=max(wanted);origin=None;received=0
    with env['av'].open(str(path)) as c:
        stream=c.streams.video[0];stream.codec_context.thread_count=1
        for index,frame in enumerate(c.decode(stream)):
            if origin is None:origin=float(frame.time)
            if index in wanted:
                expected=wanted[index];pts=float(frame.time)-origin
                rgb=env['canvas_bytes'](frame)
                assert abs(pts-expected['pts_s'])<1e-7
                assert hashlib.sha256(rgb).hexdigest()==expected['rgb_sha256']
                received+=1
                yield env['FramePacket'](index,pts,env['np'].frombuffer(rgb,dtype='uint8').reshape(192,256,3).copy(),index+1)
            if index>=last:break
    assert received==len(frames)


def worker(run,out):
    protocol,rows,groups=verify(run)
    auth=read(run/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(run/'execution_lock.json')
    cpu=read(run.parent/'CPU_preflight.json')
    assert cpu['status']=='passed' and cpu['protocol_sha256']==sha(run/'protocol.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1')
    env=runtime();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert torch.cuda.mem_get_info()[0]>=32*2**30 and env['psutil'].virtual_memory().available>=12*2**30
    from compression_observer import CompressionObserver
    from replay import VideoSource,replay
    from score_memory_eval import prompt
    out.mkdir(exist_ok=False)
    started=time.monotonic();answers=generated=encodes=resets=0;peak=0;cohorts=Counter()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    save(out/'loaded.json',dict(pretrained_loads=1,seconds=time.monotonic()-started))
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    with (out/'answers.jsonl').open('x',encoding='utf-8') as output,(out/'events.jsonl').open('x',encoding='utf-8') as events,\
         (out/'tokens.jsonl').open('x',encoding='utf-8') as tokens,torch.inference_mode(),env['fp32_reference']():
        def guard():
            nonlocal peak
            peak=max(peak,env['psutil'].Process().memory_info().rss)
            assert peak<20*2**30 and torch.cuda.memory_allocated()<40*2**30
            assert time.monotonic()-started<3300
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<64*2**20
        def record(q,result):
            nonlocal answers,generated
            assert set(result['arms'])==set(ARMS)
            for arm,answer in result['arms'].items():
                output.write(json.dumps(dict(id=q['id'],cohort=q['cohort'],status='ok',**answer))+'\n')
                answers+=1;generated+=len(answer['generated_ids']);cohorts[q['cohort']]+=1
            output.flush();guard()
        for g in groups:
            path=ROOT/g['cluster_media'];assert path.resolve().is_relative_to(ROOT.resolve()) and sha(path)==g['sha256']
            qs=sorted([q for q in rows if q['group']==g['id']],key=lambda q:(q['query_time_s'],q['id']))
            observer=CompressionObserver(env,model,build)
            try:
                if g['cohort']=='stream34':
                    spec=dict(source_path=g['source_path'],frames=g['frames'],media_sha256=g['sha256'],duration_s=g['duration_s'])
                    source=VideoSource(path,spec,env['FramePacket'])
                    # The scheduler is unchanged. Only the readout dispatch changes.
                    snapshots=read(run/'snapshots34.json')
                    def on_event(item):
                        nonlocal encodes
                        if item['event']['kind']=='query':
                            q=next(q for q in qs if q['id']==item['event']['query_id'])
                            record(q,item['answer'])
                        events.write(json.dumps(item)+'\n');events.flush();guard()
                    replay(source,qs,observer,snapshots,on_event,
                        lambda t:tokens.write(json.dumps(t)+'\n'),read_from='compression_temporal')
                else:
                    pending=None
                    for packet in packet_stream(env,path,g['frames']):
                        if pending is None:pending=packet
                        else:observer.append_pair(pending,packet);pending=None;guard()
                    if pending is not None:
                        observer.snapshot_tail(pending);encodes+=1
                    for q in qs:
                        assert len(observer.archive.nodes['full48'])==q['pairs']
                        result=observer.ask(prompt(q),read_from='compression_temporal',
                            on_token=lambda t:tokens.write(json.dumps(dict(query_id=q['id'],**t))+'\n'))
                        record(q,result)
                encodes+=observer.core.vision_encodings;resets+=observer.core.resets
                assert max(observer.archive.status()['tensor_bytes'].values())<128*2**20
                events.write(json.dumps(dict(kind='group_end',group=g['id'],archive=observer.archive.status(),
                    reduction_seconds=observer.archive.reduction_seconds,main=observer.status()))+'\n')
            finally:observer.close()
            del observer;gc.collect();torch.cuda.empty_cache();guard()
            print(dict(groups_complete=g['id'],answers=answers),flush=True)
    assert answers==909 and generated<=7272 and encodes==protocol['vision_encodings']
    save(out/'completed.json',dict(status='complete',answers=answers,generated_tokens=generated,cohorts=dict(cohorts),
        vision_encodings=encodes,main_resets=resets,pretrained_loads=1,seconds=time.monotonic()-started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),observed_RSS_bytes=peak,
        finite_diagnostic_archives_not_bounded_memory=True,wallclock_1x=False,
        shared_process_timings_not_isolated_end_to_end_speed=True,execution_lock_sha256=sha(run/'execution_lock.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();worker(a.run,a.out)
