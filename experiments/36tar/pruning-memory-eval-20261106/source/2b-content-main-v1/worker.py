"""Exactly one independently loaded process for a frozen quality shard or cost arm."""
import os,time,json,argparse,gc
from pathlib import Path
from common import ROOT,PREP,Session,MODES,runtime,DATA,make_builder,CausalSource,PairDriver,prompt,read,save,sha,verify,pack
def main(task_index,out,deadline):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm allocation required')
    verify();auth=read(PREP/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    task=read(PREP/'tasks.json')[task_index];quality=task['kind']=='quality';modes=task['modes']
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'runs').resolve())
    qs=[q for q in read(PREP/'requests.json') if q['video_id'] in task['videos']]
    expected=read(PREP/'expected_prefixes.json');videos={v['video_id']:v for v in read(PREP/'videos.json')}
    refs=read(PREP/'references.json');protocol=read(PREP/'protocol.json')
    start=time.monotonic()
    for vid in task['videos']:
        v=videos[vid];p=(DATA/v['relative_path']).resolve()
        assert p.is_relative_to(DATA.resolve()) and p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
    integrity=time.monotonic()-start
    env=runtime();t=env['torch'];t.set_num_threads(1)
    assert t.cuda.device_count()==1 and t.cuda.mem_get_info()[0]>=24*2**30
    assert env['psutil'].virtual_memory().available>=12*2**30
    out.mkdir();answers=generated=vision=warmups=matches=0;profiles=[];start=time.monotonic()
    try:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval();model.requires_grad_(False);assert len(model.model.visual.blocks)==24
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),(288,384))
        load=time.monotonic()-start;last_guard=0.
        def guard():
            nonlocal last_guard
            if time.time()>=deadline:raise TimeoutError('fixed allocation deadline')
            if time.monotonic()-last_guard<5:return
            last_guard=time.monotonic()
            assert env['psutil'].Process().memory_info().rss<20*2**30 and t.cuda.max_memory_allocated()<32*2**30
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<96*2**20
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for si,vid in enumerate(task['videos']):
                scheduled=sorted((q for q in qs if q['video_id']==vid),key=lambda q:(q['question_time'],q['question_id']))
                # Warmup uses only the first two seconds once per process, then resets and decodes anew.
                # No result from warmup enters the timed source or later visual scorer.
                if not warmups:
                    warm=Session(env,model,builder,modes=modes)
                    driver=PairDriver(CausalSource(DATA/videos[vid]['relative_path']),warm.append_pair)
                    try:
                        driver.advance(2.);assert warm.pairs==1
                        for mode in modes:
                            r=warm.ask(mode,prompt(scheduled[0]),lambda item:None)
                            generated+=len(r['generated_ids']);warmups+=1
                        warm_vision=warm.visual_calls
                    finally:driver.close();warm.close()
                    del warm;gc.collect();t.cuda.empty_cache();t.cuda.reset_peak_memory_stats()
                session=Session(env,model,builder,modes=modes);source=CausalSource(DATA/videos[vid]['relative_path'])
                def commit(a,b,*,padding):guard();return session.append_pair(a,b,padding=padding)
                driver=PairDriver(source,commit);began=time.monotonic();qa_total=0.
                try:
                    for qi,q in enumerate(scheduled):
                        state=driver.advance(q['question_time'])
                        for key in protocol['original_prefix_identity_keys']:assert state[key]==expected[q['question_id']][key],(q['question_id'],key)
                        offset=(si+qi)%len(modes);order=modes[offset:]+modes[:offset]
                        for mode in order:
                            guard();result=session.ask(mode,prompt(q),lambda item:None)
                            reference=refs.get(q['question_id']+'|'+mode)
                            if reference:
                                for key in reference:assert result[key]==reference[key],(q['question_id'],mode,key)
                                matches+=1
                            row=pack(q,state,result)
                            stream.write(json.dumps(row)+'\n');stream.flush()
                            answers+=1;generated+=len(result['generated_ids']);qa_total+=result['QA_total_seconds']
                    elapsed=time.monotonic()-began
                    profiles.append(dict(video_id=vid,total_wall_seconds=elapsed,source=source.status(),
                        session=session.status(),timing=session.timing.copy(),QA_seconds=qa_total,visual_calls=session.visual_calls))
                    vision+=session.visual_calls
                finally:driver.close();session.close()
                del session;gc.collect();t.cuda.empty_cache()
                print(json.dumps(dict(task=task['name'],sources=len(profiles),answers=answers,vision=vision)),flush=True)
        assert answers==len(qs)*len(modes) and generated<=(answers+warmups)*16
        if quality:assert matches==len(qs)*3
        save(out/'completed.json',dict(status='complete',task=task,answers=answers,warmup_answers=warmups,
            generated_tokens=generated,vision_calls=vision,warmup_vision_calls=warm_vision,baseline_matches=matches,
            profiles=profiles,pretrained_loads=1,load_seconds=load,integrity_seconds=integrity,
            peak_GPU_allocated_bytes=t.cuda.max_memory_allocated(),RSS_bytes=env['psutil'].Process().memory_info().rss,
            cost_independent_process=not quality,not_edge_or_wallclock_1x=True))
    except Exception as e:save(out/'failed.json',dict(error=str(e),answers=answers,generated_tokens=generated));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args();main(a.task,a.out,a.deadline)
