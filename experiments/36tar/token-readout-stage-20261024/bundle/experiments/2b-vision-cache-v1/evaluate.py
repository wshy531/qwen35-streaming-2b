"""One authorized two-arm run; never submits/retries/downloads a job."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
from common import ROOT, OUT, ARMS, read, save, sha, runtime


def verify(run):
    lock = read(run/'execution_lock.json')
    for name,digest in lock['input_SHA256'].items():
        assert sha(run/name)==digest,name
    for name,digest in lock['runtime_SHA256'].items():
        path=ROOT/name
        if name.startswith('repo/') and not (ROOT/'repo').exists():
            path=ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(path)==digest,name
    return read(run/'protocol.json')


def worker(run, out):
    protocol=verify(run)
    auth=read(run/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_SHA256']==sha(run/'execution_lock.json')
    cpu=read(OUT/'cluster_CPU_preflight.json')
    assert cpu['status']=='passed' and cpu['layouts']==606 and cpu['protocol_SHA256']==sha(run/'protocol.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1')
    env=runtime();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(317)
    assert torch.cuda.mem_get_info()[0]>=24*2**30 and env['psutil'].virtual_memory().available>=12*2**30
    from cacher import PairVisionCache
    from io_source import BoundedAVSource
    from session import Session
    from replay import replay
    from score_memory_eval import prompt
    out.mkdir(exist_ok=False)
    began=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),
        canvas=(192,256),resize_mode='letterbox')
    groups,questions=read(run/'groups.json'),read(run/'questions.json')
    warm=next(g for g in sorted(groups,key=lambda g:g['sha256']) if g['cohort']=='T98' and len(g['frames'])>=16)
    source=BoundedAVSource(ROOT/warm['cluster_media'],dict(warm,frames=warm['frames'][:16]),env['FramePacket'],config='auto4')
    # Bounded warmup only:16 RGB frames, not a formal/full video archive.
    packets=list(source.packets());assert len(packets)==16
    errors=[]
    with torch.inference_mode(),env['fp32_reference']():
        accept=PairVisionCache(model.model.visual,update_ratio=1.0)
        with accept:
            for first,second in (packets[:2],packets[2:4]):
                data,_=builder.pair(first,second,0,0)
                data=data.to('cuda')
                with accept.bypass():
                    reference=torch.cat(model.model.get_video_features(data.pixel_values_videos,data.video_grid_thw,return_dict=True).pooler_output,0)
                actual=torch.cat(model.model.get_video_features(data.pixel_values_videos,data.video_grid_thw,return_dict=True).pooler_output,0)
                error=float((reference-actual).abs().max())
                assert torch.allclose(reference,actual,**protocol['numerical_acceptance']),error
                errors.append(error)
        with PairVisionCache(model.model.visual) as cache:
            session=Session(env,model,builder,cache)
            for i in range(0,16,2):session.append_pair(*packets[i:i+2])
            warm_answers=session.ask('What is visible?',read_from='visual_cache',on_token=lambda t:None)
            warm_generated=sum(len(a['generated_ids']) for a in warm_answers['arms'].values())
            session.close()
    del packets,reference,actual,data,session
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    save(out/'startup.json',dict(pretrained_loads=1,numerical_max_abs_errors=errors,
        numerical_visual_calls=4,warmup_visual_calls=16,warmup_answers=2,warmup_generated_tokens=warm_generated,
        startup_load_acceptance_warmup_seconds=time.monotonic()-began))
    model_baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    count=generated=visual_calls=0;profiles=[]
    measured=time.monotonic()
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers, \
         (out/'tokens.jsonl').open('x',encoding='utf-8') as tokens, \
         (out/'events.jsonl').open('x',encoding='utf-8') as events, \
         torch.inference_mode(),env['fp32_reference'](),PairVisionCache(model.model.visual) as cache:
        def guard():
            assert time.monotonic()-began<2400
            assert env['psutil'].Process().memory_info().rss<20*2**30
            assert torch.cuda.memory_allocated()<32*2**30
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<64*2**20
        def token(item):
            tokens.write(json.dumps(item)+'\n')
        def record(q,result,committed=None):
            nonlocal count,generated
            assert set(result['arms'])==set(ARMS) and result['visual_reference_unchanged_by_QA']
            for arm,answer in result['arms'].items():
                answers.write(json.dumps(dict(id=q['id'],group=q['group'],cohort=q['cohort'],
                    status='ok',committed_frames_sha256=committed,**answer))+'\n')
                count+=1;generated+=len(answer['generated_ids'])
            answers.flush();tokens.flush();guard()
        for group in groups:
            cache.reset()
            path=(ROOT/group['cluster_media']).resolve()
            assert path.is_relative_to(ROOT.resolve())
            qs=sorted([q for q in questions if q['group']==group['id']],key=lambda q:(q['query_time_s'],q['id']))
            session=Session(env,model,builder,cache)
            started=time.monotonic()
            source=BoundedAVSource(path,group,env['FramePacket'],config='auto4',guard=guard)
            try:
                if group['cohort']=='stream34':
                    def on_event(item):
                        if item['event']['kind']=='query':
                            q=next(q for q in qs if q['id']==item['event']['query_id'])
                            record(q,item['answer'],item['committed_frames_sha256'])
                        # No feature arrays or full frame archive in the log.
                        events.write(json.dumps({k:v for k,v in item.items() if k!='answer'})+'\n')
                    replay(source,qs,session,read(run/'snapshots34.json'),on_event,token,read_from='visual_cache')
                else:
                    pending=None
                    for packet in source.packets():
                        if pending is None:pending=packet
                        else:session.append_pair(pending,packet);pending=None
                    if pending is not None:session.snapshot_tail(pending)
                    for q in qs:
                        assert session.pairs==q['pairs']
                        record(q,session.ask(prompt(q),read_from='visual_cache',
                            on_token=lambda t:token(dict(query_id=q['id'],**t))))
                torch.cuda.synchronize()
                assert cache.reference_bytes()==75497472
                profiles.append(dict(group=group['id'],cohort=group['cohort'],
                    shared_pipeline_wall_seconds=time.monotonic()-started,
                    source=source.status(),pairs=session.pairs,
                    per_arm_encoding={a:s.timing.copy() for a,s in session.sessions.items()},
                    per_arm_bank={a:s.bank.status() for a,s in session.sessions.items()},
                    full_reference_pairs=cache.full_pairs,partial_reference_pairs=cache.selective_pairs,
                    reference_bytes=cache.reference_bytes()))
                visual_calls+=session.vision_encodings
            finally:
                source.close();session.close();del session
                gc.collect();torch.cuda.empty_cache();guard()
            print(dict(groups=len(profiles),answers=count,visual_calls=visual_calls),flush=True)
    assert count==606 and generated<=4848 and visual_calls==7962
    save(out/'completed.json',dict(status='complete',formal_answers=count,formal_generated_tokens=generated,
        formal_visual_calls=visual_calls,pretrained_loads=1,profiles=profiles,
        model_baseline_GPU_allocated_bytes=model_baseline,
        overall_two_arm_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        overall_two_arm_peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
        measured_shared_process_wall_seconds=time.monotonic()-measured,
        timing_is_not_standalone_per_arm=True,not_edge_or_wallclock1x=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();worker(args.run,args.out)
