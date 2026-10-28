"""One explicitly allocated GPU worker; banks and reference in separate processes."""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import time
from sember_common import ROOT,OUT,DATA,ARMS,read,save,sha,runtime
from sember_protocol import prompt
from sember_source import CausalSource,PairDriver
from sember_runtime import make_builder
from sember_session import Session


def main(mode,out):
    # Refuse real model loading before a Slurm allocation is present.
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicitly approved Slurm GPU allocation required')
    if not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('project result directory required')
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,digest in lock['runtime_sha256'].items():
        path=ROOT/rel if not rel.startswith('repo/') or (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for name,digest in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==digest,name
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>=24*2**30
    assert env['psutil'].virtual_memory().available>=12*2**30
    protocol=read(OUT/'inputs/profile_protocol.json');requests=read(OUT/'inputs/requests.json')
    expected=read(OUT/'inputs/expected_prefixes.json');videos=read(OUT/'inputs/videos.json')
    for v in videos:
        path=(DATA/v['relative_path']).resolve();assert path.is_relative_to(DATA.resolve())
        assert path.stat().st_size==v['bytes'] and sha(path)==v['sha256']
    out.mkdir(exist_ok=False);started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    load_seconds=time.monotonic()-started
    arms=ARMS if mode=='bounded' else ('R_all_history_stream',)
    source=CausalSource(DATA/('videos/'+requests[0]['video_id']+'.mp4'))
    warm=Session(env,model,builder,mode=mode);driver=PairDriver(source,warm.append_pair)
    warm_count=0
    try:
        driver.advance(2.);assert warm.pairs==1
        for arm in arms:
            row=warm.ask(arm,'What is visible?',lambda x:None);warm_count+=len(row['generated_ids'])
    finally:warm.close();driver.close()
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    save(out/'startup.json',dict(mode=mode,load_seconds=load_seconds,warmup_answers=len(arms),
        warmup_generated_tokens=warm_count,warmup_vision_encodings=1,pretrained_loads=1))
    groups=defaultdict(list)
    for q in requests:groups[q['video_id']].append(q)
    count=generated=encodings=0;profiles=[];measured=time.monotonic()
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens:
        import json
        for vid,queries in groups.items():
            session=Session(env,model,builder,mode=mode);source=CausalSource(DATA/('videos/'+vid+'.mp4'))
            driver=PairDriver(source,session.append_pair);begin=time.monotonic()
            try:
                for index,q in enumerate(sorted(queries,key=lambda q:(q['question_time'],q['question_id']))):
                    prefix=driver.advance(q['question_time']);target=expected[q['question_id']]
                    for key in ('pairs','real_frames','last_committed_pts','submitted_prefix_sha256','true_EOF_tail_padded'):
                        assert prefix[key]==target[key],(q['question_id'],key)
                    order=arms[index%len(arms):]+arms[:index%len(arms)]
                    for arm in order:
                        def emit(item):
                            tokens.write(json.dumps(dict(question_id=q['question_id'],arm=arm,**item))+'\n')
                        row=session.ask(arm,prompt(q),emit)
                        answers.write(json.dumps(dict(question_id=q['question_id'],video_id=vid,query_prefix=prefix,**row))+'\n');answers.flush()
                        count+=1;generated+=len(row['generated_ids'])
                        assert time.monotonic()-started<1650 and torch.cuda.memory_allocated()<48*2**30
                        assert env['psutil'].Process().memory_info().rss<20*2**30
                        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<protocol['logs_limit_bytes']
                profiles.append(dict(video_id=vid,real_vision_encodings=session.pairs,source=source.status(),
                    state=session.status(),timing=session.timing.copy(),source_pipeline_wall_seconds=time.monotonic()-begin))
                encodings+=session.pairs
            finally:driver.close();session.close()
            del session;gc.collect();torch.cuda.empty_cache()
            print(dict(mode=mode,completed_sources=len(profiles),answers=count),flush=True)
    assert count==8*len(arms) and encodings==protocol['pairs_per_process']
    assert generated+warm_count<=(8*len(arms)+len(arms))*16
    save(out/'completed.json',dict(status='complete',mode=mode,answers=count,generated_tokens=generated,
        pretrained_loads=1,vision_encodings=encodings,warmup_answers=len(arms),warmup_generated_tokens=warm_count,
        wall_seconds=time.monotonic()-measured,profiles=profiles,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),shared_banks_not_standalone_cost=mode=='bounded',
        no_full_feature_archive=True,not_wallclock_1x=True,execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['bounded','reference'],required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();main(args.mode,args.out)
