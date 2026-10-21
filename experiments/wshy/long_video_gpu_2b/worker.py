"""One FP32 load, single visual forward per arrived pair, synchronous readouts."""
import gc
import json
import os
import subprocess
import sys
import time

from ..online_memory_2b.isolation import configure,output_path
configure()
from ..online_memory_2b.frozen import REPO,runtime
from ..online_memory_2b.protocol import sha,save,safe_path,verify_model
from ..online_memory_2b.runner import environment,isolated_readout
from ..online_memory_2b.receipts import successful_counts
from ..long_video_2b.protocol import validate_queries
from ..long_video_2b.scheduler import decode_samples
from ..long_video_2b.driver import run_sessions
from .gates import verify_spec,Budget,check_query


def gpu(run,spec_path):
    spec,protocol,proof,(queries,sources,snapshots)=verify_spec(run,spec_path)
    run=output_path(run)
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','-1')
            or os.environ.get('SLURM_CPUS_PER_TASK')!='4' or int(os.environ.get('SLURM_RESTART_COUNT','0'))!=0):
        raise ValueError('approved one-GPU allocation, four CPUs and no restart required')
    verify_model(protocol['model'])
    out=output_path(run/'gpu')
    out.mkdir(exist_ok=False)
    # New extension contracts run in a child with GPU hidden before any weights.
    test_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    log=out/'extension_contracts.txt'
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',
            str(REPO/'experiments/wshy/long_video_gpu_2b/tests'),'-v'],cwd=REPO,env=test_env,
            stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    if result.returncode:
        raise RuntimeError('GPU extension contracts failed before model load')
    successful_counts(log.read_text(encoding='utf-8'),[spec['extension_contract_tests']])
    # Contracts must not have altered any approved source, protocol or proof.
    verify_spec(run,spec_path)
    env=runtime()
    torch=env['torch']
    if torch.cuda.device_count()!=1 or environment(env)!=proof['environment']:
        raise ValueError('GPU allocation or CPU/GPU software identity mismatch')
    torch.set_num_threads(1)
    torch.manual_seed(123)
    free,_=torch.cuda.mem_get_info()
    if free<16*2**30 or env['psutil'].virtual_memory().available<12*2**30:
        raise MemoryError('preload resource gate')
    save(out/'guard.json',dict(job=os.environ['SLURM_JOB_ID'],GPU=torch.cuda.get_device_name(),
        free_GPU_bytes=free,protocol_sha256=sha(spec_path),CPU_proof_sha256=sha(run/'cpu/preflight.json')))
    save(out/'load_started.json',dict(maximum_pretrained_loads=1,automatic_retry=False))
    started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(protocol['model'],local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    if (model.config.text_config.hidden_size!=2048
            or any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters())):
        raise ValueError('FP32 2B model identity required')
    save(out/'loaded.json',dict(pretrained_loads=1,seconds=time.monotonic()-started))
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(protocol['model'],local_files_only=True),
        canvas=(192,256),resize_mode='letterbox')
    budget=Budget()
    vision_calls=encodes=peak_rss=peak_bank=0
    timings=dict(decode=0.,encode=0.,answer=0.)
    snap_byid={s['query_id']:s for s in snapshots}
    groups=validate_queries(queries,canonical=True)
    summaries=[]

    def guard():
        nonlocal peak_rss
        import resource
        peak_rss=max(peak_rss,env['psutil'].Process().memory_info().rss,
                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        log_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        budget.resources(time.monotonic()-started,peak_rss,torch.cuda.max_memory_allocated(),log_bytes)

    def hook(*unused):
        nonlocal vision_calls
        vision_calls+=1

    handle=model.model.visual.register_forward_hook(hook)
    try:
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as predictions, \
             (out/'events.jsonl').open('x',encoding='utf-8') as events, \
             torch.inference_mode(),env['fp32_reference']():
            def write(stream,row):
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush()

            def readout(nodes,question):
                guard()
                torch.cuda.synchronize()
                tick=time.monotonic()
                result=isolated_readout(env,model,builder,nodes,question)
                torch.cuda.synchronize()
                timings['answer']+=time.monotonic()-tick
                if vision_calls!=encodes:
                    raise RuntimeError('QA unexpectedly encoded visual input')
                return result

            for source in sources:
                guard()
                path=safe_path(protocol['resource_root'],source['media_path'])
                if sha(path)!=source['media_sha256']:
                    raise ValueError('media changed since CPU preflight')
                bank_bytes={}

                def encode(index,first,second):
                    nonlocal encodes
                    guard()
                    torch.cuda.synchronize()
                    tick=time.monotonic()
                    inputs,_=builder.pair(first,second,0,0,synthetic_padding=False)
                    features=torch.cat(model.model.get_video_features(inputs.pixel_values_videos.to('cuda'),
                        inputs.video_grid_thw.to('cuda'),return_dict=True).pooler_output,0)
                    block=env['legacy'].features.raw_block(features,index,first.timestamp,second.timestamp)
                    del inputs,features
                    torch.cuda.synchronize()
                    timings['encode']+=time.monotonic()-tick
                    encodes+=1
                    if vision_calls!=encodes or encodes>1360:
                        raise RuntimeError('repeated, missing or excess visual encoding')
                    return block

                def event(row):
                    nonlocal peak_bank
                    if row['event']=='append':
                        state=row['state']
                        if (not state['valid'] or state['closed'] or state['visual_tokens']>384
                                or len(state['nodes'])>8 or state['resident_bytes']>8*394560+8192
                                or state['metadata_bytes']>32768 or state['full_history_retained']
                                or state['language_cache_retained']):
                            raise RuntimeError('visual bank memory/isolation bound exceeded')
                        bank_bytes[row['arm']]=state['resident_bytes']
                    elif row['event']=='query':
                        check_query(snap_byid[row['id']],row)
                    else:
                        raise ValueError('unknown driver event')
                    write(events,dict(source_path=source['source_path'],**row))
                    if row['event']=='append' and row['arm']=='reservoir4':
                        combined=sum(bank_bytes.values())
                        peak_bank=max(peak_bank,combined)
                        write(events,dict(event='resources',source_path=source['source_path'],pair=row['pair'],
                            combined_bank_bytes=combined,GPU_allocated=torch.cuda.memory_allocated(),
                            GPU_reserved=torch.cuda.memory_reserved(),RSS_bytes=env['psutil'].Process().memory_info().rss))
                    guard()

                def answer(row):
                    check_query(snap_byid[row['id']],row['prefix'])
                    budget.answer(row)
                    write(predictions,row)
                    guard()

                def samples():
                    iterator=decode_samples(env,path,source['frames'])
                    try:
                        while True:
                            tick=time.monotonic()
                            try:
                                packet=next(iterator)
                            except StopIteration:
                                return
                            timings['decode']+=time.monotonic()-tick
                            yield packet
                            del packet
                    finally:
                        iterator.close()
                summary=run_sessions(source['frames'],groups[source['source_path']],samples(),encode,
                    readout,env['prompt'],event,answer)
                summaries.append(dict(source_path=source['source_path'],**summary))
                # Hashing is bounded I/O after all questions, no future decode.
                if sha(path)!=source['media_sha256']:
                    raise ValueError('media changed during execution')
                gc.collect()
                guard()
                print(json.dumps(dict(sources=len(summaries),answers=budget.answers,
                    generated_tokens=budget.tokens,visual_forwards=vision_calls)),flush=True)
            if (budget.answers!=204 or encodes!=1360 or vision_calls!=1360
                    or sum(s['sampled_frames'] for s in summaries)!=2726
                    or sum(s['reset_checks'] for s in summaries)!=30):
                raise RuntimeError('complete six-source/34-query coverage required')
            guard()
            save(out/'completed.json',dict(status='complete',pretrained_loads=1,main_answers=budget.answers,
                generated_tokens=budget.tokens,source_pairs_encoded=encodes,visual_forward_calls=vision_calls,
                reset_checks=30,all_QA_banks_unchanged=True,sources=summaries,
                peak_combined_bank_bytes=peak_bank,peak_RSS_bytes=peak_rss,
                peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
                seconds=time.monotonic()-started,stage_seconds=timings,environment=environment(env),
                GPU_protocol_sha256=sha(spec_path),CPU_preflight_sha256=sha(run/'cpu/preflight.json'),
                extension_contract_log_sha256=sha(log),extension_contract_tests=spec['extension_contract_tests'],
                output_sha256={name:sha(out/name) for name in ('predictions.jsonl','events.jsonl')},
                independent_content_review='pending',historical_answer_identity_claim=False))
    finally:
        handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
