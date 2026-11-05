"""Fixed full comparison; input/control core unchanged from validated profile."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time
from main_common import ROOT,OUT,DATA,ARMS,read,save,sha,runtime,verify
from sember_protocol import prompt
from sember_source import CausalSource,PairDriver
from sember_runtime import make_builder
from sember_session import Session


def group_queries(requests):
    grouped=defaultdict(list)
    for q in requests:grouped[q['video_id']].append(q)
    return [(vid,sorted(qs,key=lambda q:(q['question_time'],q['question_id']))) for vid,qs in sorted(grouped.items())]


def main(mode,out,deadline):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicitly approved Slurm GPU allocation required')
    if not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('project result directory required')
    protocol=verify();requests=read(OUT/'inputs/requests.json');expected=read(OUT/'inputs/expected_prefixes.json')
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>=24*2**30
    assert env['psutil'].virtual_memory().available>=12*2**30
    # The media receipt is separately verified before allocation; no hidden fallback download.
    receipt=read(DATA/'formal128_transfer_receipt.json');assert receipt['status']=='complete'
    integrity_started=time.monotonic()
    for v in read(OUT/'inputs/videos.json'):
        path=(DATA/v['relative_path']).resolve();assert path.is_relative_to(DATA.resolve())
        assert path.stat().st_size==v['bytes'] and sha(path)==v['sha256'],v['video_id']
    integrity_seconds=time.monotonic()-integrity_started
    out.mkdir(exist_ok=False);started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    arms=ARMS if mode=='bounded' else (protocol['reference_arm'],);load_seconds=time.monotonic()-started
    warm=Session(env,model,builder,mode=mode);source=CausalSource(DATA/('videos/'+protocol['warmup']['video_id']+'.mp4'))
    driver=PairDriver(source,warm.append_pair);warm_generated=0
    try:
        driver.advance(protocol['warmup']['query_time']);assert warm.pairs==1
        for arm in arms:warm_generated+=len(warm.ask(arm,'What is visible?',lambda item:None)['generated_ids'])
    finally:driver.close();warm.close()
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    save(out/'startup.json',dict(mode=mode,pretrained_loads=1,load_seconds=load_seconds,
        warmup_answers=len(arms),warmup_generated_tokens=warm_generated,warmup_encodings=1,
        media_integrity_seconds=integrity_seconds,
        model_baseline_allocated_bytes=baseline))
    count=generated=encodings=0;profiles=[];began=time.monotonic();query_serial=0;last_resource_check=0.
    def guard():
        nonlocal last_resource_check
        if time.time()>=deadline:raise TimeoutError('approved shared walltime reached; no auto continuation')
        if time.monotonic()-last_resource_check<5.:return
        last_resource_check=time.monotonic()
        assert env['psutil'].Process().memory_info().rss<protocol['worker_RSS_limit_bytes']
        assert torch.cuda.max_memory_allocated()<protocol['GPU_allocated_limit_bytes']
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())<protocol['log_limit_bytes']
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens:
        for vid,queries in group_queries(requests):
            session=Session(env,model,builder,mode=mode);source=CausalSource(DATA/('videos/'+vid+'.mp4'))
            def append(a,b,*,padding):
                guard();return session.append_pair(a,b,padding=padding)
            driver=PairDriver(source,append);source_begin=time.monotonic()
            try:
                for q in queries:
                    prefix=driver.advance(q['question_time']);target=expected[q['question_id']]
                    for key in ('pairs','real_frames','last_committed_pts','submitted_prefix_sha256','true_EOF_tail_padded'):
                        assert prefix[key]==target[key],(q['question_id'],key)
                    offset=query_serial%len(arms);order=arms[offset:]+arms[:offset]
                    for arm in order:
                        guard()
                        def emit(item):tokens.write(json.dumps(dict(question_id=q['question_id'],arm=arm,**item))+'\n')
                        row=session.ask(arm,prompt(q),emit)
                        answers.write(json.dumps(dict(question_id=q['question_id'],video_id=vid,query_prefix=prefix,**row))+'\n');answers.flush();tokens.flush()
                        count+=1;generated+=len(row['generated_ids'])
                    query_serial+=1
                profiles.append(dict(video_id=vid,vision_encodings=session.pairs,source=source.status(),
                    state=session.status(),timing=session.timing.copy(),source_wall_seconds=time.monotonic()-source_begin))
                encodings+=session.pairs
                with (out/'source_checkpoints.jsonl').open('a',encoding='utf-8') as checkpoints:
                    checkpoints.write(json.dumps(dict(video_id=vid,source_index=len(profiles),answers=count,
                        generated_tokens=generated,vision_encodings=encodings,wall_seconds=time.monotonic()-began))+'\n')
            finally:driver.close();session.close()
            del session;gc.collect();torch.cuda.empty_cache()
            print(dict(mode=mode,completed_sources=len(profiles),answers=count,encodings=encodings),flush=True)
    assert count==protocol['questions']*len(arms) and encodings==protocol['pairs_per_process']
    assert generated+warm_generated<=(protocol['questions']+1)*len(arms)*16
    save(out/'completed.json',dict(status='complete',mode=mode,answers=count,generated_tokens=generated,
        warmup_answers=len(arms),warmup_generated_tokens=warm_generated,warmup_encodings=1,
        vision_encodings=encodings,pretrained_loads=1,wall_seconds=time.monotonic()-began,profiles=profiles,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        last_RSS_bytes=env['psutil'].Process().memory_info().rss,shared_costs_not_standalone=mode=='bounded',
        source_checkpoint_is_not_automatic_resume=True,not_wallclock_1x=True,execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['bounded','reference'],required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();main(args.mode,args.out,args.deadline)
