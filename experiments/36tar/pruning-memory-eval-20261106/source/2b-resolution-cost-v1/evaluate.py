"""Independent sequential FP32 processes; causal file replay and complete cost ledger."""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import time
from resolution_common import ROOT,OUT,MAIN,DATA,SPECS,read,save,sha,runtime,verify
from resolution_source import CausalSource,PairDriver
from resolution_runtime import make_builder
from resolution_session import Session
from sember_protocol import prompt


def readlines(path):
    import json
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def main(task,out,deadline,run_root):
    import json
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('approved Slurm allocation required')
    protocol=verify();canvas=tuple(task['canvas']);arms=tuple(task['arms'])
    requests=read(OUT/'inputs/requests.json');expected=read(OUT/'inputs/expected_mid_prefixes.json')
    grouped=defaultdict(list)
    for q in requests:
        if task['videos'] is None or q['video_id'] in task['videos']:grouped[q['video_id']].append(q)
    videos={v['video_id']:v for v in read(OUT/'inputs/videos.json')}
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>=24*2**30
    assert env['psutil'].virtual_memory().available>=12*2**30
    out.mkdir(exist_ok=False)
    integrity_started=time.monotonic()
    for vid in grouped:
        v=videos[vid];p=(DATA/v['relative_path']).resolve();assert p.is_relative_to(DATA.resolve())
        assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],vid
    integrity_seconds=time.monotonic()-integrity_started
    started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor,canvas);load_seconds=time.monotonic()-started
    warm=Session(env,model,builder,arms=arms)
    driver=PairDriver(CausalSource(DATA/('videos/'+protocol['warmup']['video_id']+'.mp4'),canvas=canvas),warm.append_pair)
    warm_generated=0
    try:
        driver.advance(protocol['warmup']['query_time']);assert warm.pairs==1
        for a in arms:warm_generated+=len(warm.ask(a,'What is visible?',lambda item:None)['generated_ids'])
    finally:driver.close();warm.close()
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    save(out/'startup.json',dict(task=task,load_seconds=load_seconds,integrity_seconds=integrity_seconds,
        pretrained_loads=1,warmup_answers=len(arms),warmup_generated_tokens=warm_generated,warmup_encodings=1))
    prior={}
    if task['kind']=='quality' and canvas==(288,384):
        rows=readlines(ROOT/'runs/2b-sember-main-218526/bounded/answers.jsonl')
        mapping={p+'_full108'+('_8' if p=='M0' else ''):p+'_mid108' for p in ('M0','M2','A06','LS')}
        prior={(r['question_id'],mapping[r['arm']]):r for r in rows if r['arm'] in mapping}
        assert len(prior)==478*4
    elif task['kind']=='cost':
        name='quality_mid' if canvas==(288,384) else 'quality_high'
        prior={(r['question_id'],r['arm']):r for r in readlines(run_root/name/'answers.jsonl') if r['arm'] in arms}
    count=generated=encodings=0;profiles=[];began=time.monotonic();last_check=0.;serial=0
    def guard():
        nonlocal last_check
        if time.time()>=deadline:raise TimeoutError('shared allocation deadline; no auto continuation')
        if time.monotonic()-last_check<5:return
        last_check=time.monotonic()
        assert env['psutil'].Process().memory_info().rss<protocol['worker_RSS_limit_bytes']
        assert torch.cuda.max_memory_allocated()<protocol['GPU_allocated_limit_bytes']
        assert sum(p.stat().st_size for p in run_root.rglob('*') if p.is_file())<protocol['log_limit_bytes']
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens:
        for vid,queries in sorted(grouped.items()):
            guard();session=Session(env,model,builder,arms=arms)
            source=CausalSource(DATA/('videos/'+vid+'.mp4'),canvas=canvas)
            def append(a,b,*,padding):guard();return session.append_pair(a,b,padding=padding)
            driver=PairDriver(source,append);source_started=time.monotonic();qa_total=audit_total=0.
            try:
                for q in sorted(queries,key=lambda q:(q['question_time'],q['question_id'])):
                    prefix=driver.advance(q['question_time']);target=expected[q['question_id']]
                    for key in protocol['source_identity_keys']:assert prefix[key]==target[key],(q['question_id'],key)
                    if canvas==(288,384):assert prefix['submitted_prefix_sha256']==target['submitted_prefix_sha256']
                    order=arms[serial%len(arms):]+arms[:serial%len(arms)]
                    for arm in order:
                        guard()
                        def emit(item):tokens.write(json.dumps(dict(question_id=q['question_id'],arm=arm,**item))+'\n')
                        r=session.ask(arm,prompt(q),emit);old=prior.get((q['question_id'],arm))
                        if prior:
                            assert old is not None
                            for key in ('generated_ids','text','question_suffix_sha256','bounded_context_ids_sha256','visual_tokens'):
                                assert r[key]==old[key],(q['question_id'],arm,key)
                        answers.write(json.dumps(dict(question_id=q['question_id'],video_id=vid,query_prefix=prefix,**r))+'\n')
                        answers.flush();tokens.flush();count+=1;generated+=len(r['generated_ids'])
                        qa_total+=r['QA_total_seconds'];audit_total+=r['state_audit_seconds']
                    serial+=1
                profiles.append(dict(video_id=vid,source=source.status(),state=session.status(),timing=session.timing.copy(),
                    QA_seconds=qa_total,audit_seconds=audit_total,source_wall_seconds=time.monotonic()-source_started))
                encodings+=session.pairs
                with (out/'source_checkpoints.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(sources=len(profiles),answers=count,encodings=encodings))+'\n')
            finally:driver.close();session.close()
            del session;gc.collect();torch.cuda.empty_cache()
    expected_answers=sum(len(v) for v in grouped.values())*len(arms)
    assert count==expected_answers and generated+warm_generated<=(count+len(arms))*16
    save(out/'completed.json',dict(status='complete',task=task,answers=count,generated_tokens=generated,
        warmup_answers=len(arms),warmup_generated_tokens=warm_generated,warmup_encodings=1,
        vision_encodings=encodings,pretrained_loads=1,wall_seconds=time.monotonic()-began,
        profiles=profiles,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),last_RSS_bytes=env['psutil'].Process().memory_info().rss,
        independent_single_method_cost=task['kind']=='cost',
        quality_costs_shared_across_policies=task['kind']=='quality',
        audit_in_walltime=True,decode_preprocess_vision_update_QA_in_walltime=True,
        warmup_and_model_load_outside_measured_walltime=True,
        replay_answers_checked=len(prior) if prior else 0,execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    main(read(OUT/'inputs/tasks.json')[a.task],a.out,a.deadline,a.run_root)
