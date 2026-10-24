"""One process, one model load, one active method. No submission or retry."""
import argparse,gc,json,os,resource,time
from pathlib import Path
from common import ROOT,OUT,ARMS,read,save,sha,runtime


def verify(run):
    lock=read(run/'execution_lock.json');auth=read(run/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(run/'execution_lock.json')
    assert lock['protocol_sha256']==sha(run/'protocol.json')
    for name,digest in lock['input_sha256'].items():assert sha(run/name)==digest,name
    for rel,digest in lock['runtime_sha256'].items():
        path=ROOT/rel
        if rel.startswith('repo/') and not (ROOT/'repo').exists():path=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1')
    return read(run/'protocol.json')


def worker(run,out,job_index):
    protocol=verify(run);job=read(run/'jobs.json')[job_index];arm=job['arm']
    assert out.resolve().is_relative_to((ROOT/'runs').resolve()) and arm in ARMS
    cpu=read(run.parent/'CPU_preflight.json');assert cpu['status']=='passed' and cpu['protocol_sha256']==sha(run/'protocol.json')
    env=runtime();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    from session import Session
    from source import ExactPackets
    from replay import VideoSource,replay
    from score_memory_eval import prompt
    assert torch.cuda.mem_get_info()[0]>=24*2**30 and env['psutil'].virtual_memory().available>=12*2**30
    out.mkdir(exist_ok=False);started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    load_seconds=time.monotonic()-started
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    groups=read(run/'groups.json');questions=read(run/'questions.json')
    warm=next(g for g in groups if g['id']==protocol['warmup_group'])
    warm_source=ExactPackets(env,ROOT/warm['cluster_media'],warm['frames'][:16],warm['sha256'])
    warm_started=time.monotonic();session=Session(env,model,builder,arm);pending=None
    with torch.inference_mode(),env['fp32_reference']():
        for packet in warm_source.packets():
            if pending is None:pending=packet
            else:session.append_pair(pending,packet);pending=None
        warm_answer=session.ask('What is visible?',read_from='standalone',on_token=lambda t:None)['arms'][arm]
        assert session.vision_encodings==8;session.close()
    warm_generated=len(warm_answer['generated_ids']);del session;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    warm_seconds=time.monotonic()-warm_started
    save(out/'startup.json',dict(job=job,pretrained_loads=1,model_load_seconds=load_seconds,
        warmup_seconds=warm_seconds,warmup_encodings=8,warmup_answers=1,warmup_generated_tokens=warm_generated))
    torch.cuda.reset_peak_memory_stats()
    baseline=dict(GPU_allocated_bytes=torch.cuda.memory_allocated(),GPU_reserved_bytes=torch.cuda.memory_reserved(),
                  host_RSS_bytes=env['psutil'].Process().memory_info().rss)
    count=generated=encodes=0;profiles=[];cohorts={};measurement_started=time.monotonic()
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens,torch.inference_mode(),env['fp32_reference']():
        for group in [g for g in groups if g['id'] in job['groups']]:
            path=(ROOT/group['cluster_media']).resolve();assert path.is_relative_to(ROOT.resolve())
            qs=sorted([q for q in questions if q['group']==group['id']],key=lambda q:(q['query_time_s'],q['id']))
            session=Session(env,model,builder,arm);source_started=time.monotonic();io_seconds=0.;qa=[]
            def guard():
                assert time.monotonic()-started<1800
                assert env['psutil'].Process().memory_info().rss<20*2**30
                assert torch.cuda.memory_allocated()<32*2**30
            def token(item):
                nonlocal io_seconds
                start=time.monotonic();tokens.write(json.dumps(item)+'\n');io_seconds+=time.monotonic()-start
            def record(q,result,committed=None):
                nonlocal count,generated,io_seconds
                assert set(result['arms'])=={arm};answer=result['arms'][arm]
                assert answer['banks_unchanged'] and answer['branch_released']
                start=time.monotonic();answers.write(json.dumps(dict(id=q['id'],group=q['group'],cohort=q['cohort'],status='ok',
                    committed_frames_sha256=committed,**answer))+'\n');answers.flush()
                io_seconds+=time.monotonic()-start
                count+=1;generated+=len(answer['generated_ids']);cohorts[q['cohort']]=cohorts.get(q['cohort'],0)+1
                qa.append({k:answer[k] for k in ('input_tokens','visual_tokens','context_build_seconds','prefill_seconds','decode_seconds','QA_total_seconds','QA_cache_bytes')});guard()
            try:
                if group['cohort']=='stream34':
                    source=VideoSource(path,dict(source_path=group['source_path'],frames=group['frames'],media_sha256=group['sha256'],duration_s=group['duration_s']),env['FramePacket'])
                    def event(item):
                        if item['event']['kind']=='query':
                            q=next(q for q in qs if q['id']==item['event']['query_id'])
                            record(q,item['answer'],item['committed_frames_sha256'])
                    replay(source,qs,session,read(run/'snapshots34.json'),event,token,read_from='standalone')
                    source_cost=dict(decode_seconds=source.decode_seconds,transform_seconds=source.transform_seconds,
                        media_integrity_sha_seconds=source.hash_seconds,open_seconds_in_wall_unseparated=True,
                        decoded_frames=source.decoded_frames,sampled=source.sampled,peak_RGB_buffers=source.peak_buffers)
                else:
                    source=ExactPackets(env,path,group['frames'],group['sha256']);pending=None
                    for packet in source.packets():
                        if pending is None:pending=packet
                        else:session.append_pair(pending,packet);pending=None
                    if pending is not None:session.snapshot_tail(pending)
                    for q in qs:
                        assert session.pairs==q['pairs']
                        record(q,session.ask(prompt(q),read_from='standalone',on_token=lambda t:token(dict(query_id=q['id'],**t))))
                    source_cost=source.status()
                torch.cuda.synchronize();wall=time.monotonic()-source_started
                state=session.status();encodes+=session.vision_encodings
                assert set(session.bank.banks)=={arm}
                profiles.append(dict(group=group['id'],cohort=group['cohort'],arm=arm,
                    pipeline_wall_seconds=wall,source=source_cost,encoding=session.timing.copy(),
                    transfer_validate_seconds=session.bank.transfer_validate_seconds,
                    selection_seconds=session.bank.selection_seconds,bank_update_seconds=session.bank.update_seconds[arm],
                    audit_seconds=session.audit_seconds,log_write_seconds=io_seconds,
                    bank=state['banks']['banks'][arm],QA=qa,vision_encodings=session.vision_encodings))
            finally:
                session.close();del session;gc.collect();torch.cuda.empty_cache();guard()
            print(dict(job_index=job_index,arm=arm,groups=len(profiles),answers=count),flush=True)
    assert count==job['questions']
    save(out/'completed.json',dict(status='complete',arm=arm,block=job['block'],measured_answers=count,
        measured_generated_tokens=generated,measured_encodings=encodes,cohorts=cohorts,
        measured_block_wall_seconds=time.monotonic()-measurement_started,baseline=baseline,
        measured_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        measured_peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
        process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_RSS_includes_loading_and_warmup=True,profiles=profiles,
        no_other_bank=True,no_main_language_cache=True,not_wallclock_1x=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--job-index',type=int,required=True)
    args=parser.parse_args();worker(args.run,args.out,args.job_index)
