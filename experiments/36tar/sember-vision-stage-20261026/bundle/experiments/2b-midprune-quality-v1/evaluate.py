"""Four-arm causal quality run, or exactly one fully independent cost method."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time
from quality_common import ROOT,OUT,DATA,ARMS,OLD_ARMS,read,save,sha,verify,runtime,lines,golden_path
from quality_session import Session
from resolution_runtime import make_builder
from resolution_source import CausalSource,PairDriver
from sember_protocol import prompt


def main(task,out,deadline,run_root):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit new GPU approval and Slurm allocation required')
    protocol=verify();arms=tuple(task['arms']);quality=task['kind']=='quality'
    requests=read(OUT/'inputs/requests.json');expected=read(OUT/'inputs/expected_prefixes.json')
    layouts=read(OUT/'inputs/expected_layouts.json');grouped=defaultdict(list)
    for q in requests:
        if task['videos'] is None or q['video_id'] in task['videos']:grouped[q['video_id']].append(q)
    videos={v['video_id']:v for v in read(OUT/'inputs/videos.json')}
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>=24*2**30
    assert env['psutil'].virtual_memory().available>=12*2**30
    out.mkdir(exist_ok=False);integrity_started=time.monotonic()
    for vid in grouped:
        v=videos[vid];p=(DATA/v['relative_path']).resolve();assert p.is_relative_to(DATA.resolve())
        assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],vid
    assert sha(golden_path())==protocol['golden_answers_SHA']
    integrity_seconds=time.monotonic()-integrity_started;started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert len(model.model.visual.blocks)==24 and model.config.text_config.hidden_size==2048
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor,(288,384));load_seconds=time.monotonic()-started
    warm=Session(env,model,builder,arms=arms);w=protocol['warmup']
    driver=PairDriver(CausalSource(DATA/'videos'/(w['video_id']+'.mp4')),warm.append_pair);warm_generated=0
    try:
        driver.advance(w['query_time']);assert warm.pairs==1
        for arm in arms:warm_generated+=len(warm.ask(arm,'What is visible?',lambda item:None)['generated_ids'])
        warm_encodings=warm.visual_calls
    finally:driver.close();warm.close()
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    save(out/'startup.json',dict(task=task,load_seconds=load_seconds,integrity_seconds=integrity_seconds,
        pretrained_loads=1,warmup_answers=len(arms),warmup_generated_tokens=warm_generated,warmup_encodings=warm_encodings))
    prior={(r['question_id'],arm):r for r in lines(golden_path()) for arm,old in OLD_ARMS.items() if r['arm']==old} if quality else {
        (r['question_id'],r['arm']):r for r in lines(run_root/'quality/answers.jsonl') if r['arm'] in arms}
    count=generated=encodings=replayed=0;profiles=[];began=time.monotonic();last_check=0.;serial=0
    def guard():
        nonlocal last_check
        if time.time()>=deadline:raise TimeoutError('approved allocation deadline; no automatic continuation')
        if time.monotonic()-last_check<5:return
        last_check=time.monotonic()
        assert env['psutil'].Process().memory_info().rss<protocol['worker_RSS_limit_bytes']
        assert torch.cuda.max_memory_allocated()<protocol['GPU_allocated_limit_bytes']
        assert sum(p.stat().st_size for p in run_root.rglob('*') if p.is_file())<protocol['log_limit_bytes']
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens:
        for vid,queries in sorted(grouped.items()):
            guard();session=Session(env,model,builder,arms=arms);source=CausalSource(DATA/'videos'/(vid+'.mp4'))
            def append(a,b,*,padding):guard();return session.append_pair(a,b,padding=padding)
            driver=PairDriver(source,append);source_started=time.monotonic();qa_total=audit_total=0.
            try:
                for q in sorted(queries,key=lambda q:(q['question_time'],q['question_id'])):
                    prefix=driver.advance(q['question_time']);target=expected[q['question_id']]
                    for key in protocol['source_identity_keys']:assert prefix[key]==target[key],(q['question_id'],key)
                    for arm in arms[serial%len(arms):]+arms[:serial%len(arms)]:
                        guard()
                        def emit(item):tokens.write(json.dumps(dict(question_id=q['question_id'],arm=arm,**item))+'\n')
                        r=session.ask(arm,prompt(q),emit);layout=layouts[q['question_id']][arm]
                        for key in ('bounded_context_ids_sha256','visual_tokens','input_tokens'):assert r[key]==layout[key],(q['question_id'],arm,key)
                        assert [n['first'] for n in r['nodes']]==layout['source_pair_ids']
                        old=prior.get((q['question_id'],arm))
                        if old is not None:
                            for key in ('generated_ids','text','question_suffix_sha256','bounded_context_ids_sha256','visual_tokens'):
                                assert r[key]==old[key],(q['question_id'],arm,key)
                            replayed+=1
                        answers.write(json.dumps(dict(question_id=q['question_id'],video_id=vid,query_prefix=prefix,**r))+'\n')
                        answers.flush();tokens.flush();count+=1;generated+=len(r['generated_ids'])
                        qa_total+=r['QA_total_seconds'];audit_total+=r['state_audit_seconds']
                    serial+=1
                profiles.append(dict(video_id=vid,source=source.status(),state=session.status(),timing=session.timing.copy(),
                    QA_seconds=qa_total,audit_seconds=audit_total,vision_encodings=session.visual_calls,
                    source_wall_seconds=time.monotonic()-source_started))
                encodings+=session.visual_calls
                with (out/'source_checkpoints.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(sources=len(profiles),answers=count,vision_encodings=encodings))+'\n')
            finally:driver.close();session.close()
            del session;gc.collect();torch.cuda.empty_cache()
            print(dict(task=task['name'],sources=len(profiles),answers=count),flush=True)
    assert count==sum(map(len,grouped.values()))*len(arms)
    assert replayed==(1434 if quality else count)
    assert generated+warm_generated<=(count+len(arms))*16
    save(out/'completed.json',dict(status='complete',task=task,answers=count,generated_tokens=generated,
        warmup_answers=len(arms),warmup_generated_tokens=warm_generated,warmup_encodings=warm_encodings,
        vision_encodings=encodings,pretrained_loads=1,wall_seconds=time.monotonic()-began,profiles=profiles,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        last_RSS_bytes=env['psutil'].Process().memory_info().rss,load_seconds=load_seconds,integrity_seconds=integrity_seconds,
        independent_single_method_cost=not quality,quality_shared_vision_not_E2E=quality,
        replayed_answers=replayed,audits_in_walltime=True,execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    main(read(OUT/'inputs/tasks.json')[a.task],a.out,a.deadline,a.run_root)
