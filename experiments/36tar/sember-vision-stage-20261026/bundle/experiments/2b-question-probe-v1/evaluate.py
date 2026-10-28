"""One separately approved question-probe GPU run, causal inputs and no retries."""
import argparse,gc,json,os,time
from pathlib import Path
from common import ROOT,OUT,ARMS,read,save,sha,runtime

def verify(run):
    lock=read(run/'execution_lock.json');auth=read(run/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(run/'execution_lock.json')
    assert lock['protocol_sha256']==sha(run/'protocol.json')
    for name,digest in lock['input_sha256'].items():assert sha(run/name)==digest,name
    for rel,digest in lock['runtime_sha256'].items():
        p=ROOT/rel if not rel.startswith('repo/') or (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(p)==digest,rel
    cpu=read(run.parent/'CPU_preflight.json')
    assert cpu['status']=='passed' and len(cpu['layout_rows'])==909
    assert cpu['groups_sha256']==sha(run/'groups.json') and cpu['questions_sha256']==sha(run/'questions.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1')
    return read(run/'protocol.json')

def worker(run,out):
    protocol=verify(run);env=runtime();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert out.resolve().is_relative_to((ROOT/'runs').resolve())
    assert torch.cuda.mem_get_info()[0]>=24*2**30 and env['psutil'].virtual_memory().available>=12*2**30
    from session import Session
    from io_source import BoundedAVSource
    from replay import replay
    from score_memory_eval import prompt
    out.mkdir(exist_ok=False);started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    load_seconds=time.monotonic()-started
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    groups=read(run/'groups.json');rows=read(run/'questions.json')
    warm=next(g for g in groups if g['cohort']=='T98' and len(g['frames'])>=16)
    source=BoundedAVSource(ROOT/warm['cluster_media'],dict(warm,frames=warm['frames'][:16]),env['FramePacket'],config='auto4')
    warm_start=time.monotonic();s=Session(env,model,builder);pending=None
    with torch.inference_mode(),env['fp32_reference']():
        for packet in source.packets():
            if pending is None:pending=packet
            else:s.append_pair(pending,packet);pending=None
        warm_answers=s.ask('What is visible?',read_from='readout_banks',on_token=lambda t:None)
        assert s.vision_encodings==8;s.close()
    warm_generated=sum(len(a['generated_ids']) for a in warm_answers['arms'].values())
    del s,pending,packet;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    save(out/'startup.json',dict(pretrained_loads=1,load_seconds=load_seconds,warmup_seconds=time.monotonic()-warm_start,
        warmup_answers=3,warmup_probes=1,warmup_generated_tokens=warm_generated,warmup_encodings=8))
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    count=generated=encodes=0;profiles=[];measured=time.monotonic()
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers, (out/'tokens.jsonl').open('x',encoding='utf-8') as tokens, \
         torch.inference_mode(),env['fp32_reference']():
        def guard():
            assert time.monotonic()-started<2400
            assert env['psutil'].Process().memory_info().rss<20*2**30 and torch.cuda.memory_allocated()<32*2**30
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<64*2**20
        for number,g in enumerate(groups):
            path=(ROOT/g['cluster_media']).resolve();assert path.is_relative_to(ROOT.resolve())
            qs=sorted([q for q in rows if q['group']==g['id']],key=lambda q:(q['query_time_s'],q['id']))
            s=Session(env,model,builder);source=BoundedAVSource(path,g,env['FramePacket'],config='auto4',guard=guard)
            start=time.monotonic()
            def record(q,result,committed=None):
                nonlocal count,generated
                assert set(result['arms'])==set(ARMS)
                for arm,a in result['arms'].items():
                    assert a['banks_unchanged'] and a['branch_released']
                    answers.write(json.dumps(dict(id=q['id'],group=q['group'],cohort=q['cohort'],status='ok',committed_frames_sha256=committed,**a))+'\n')
                    count+=1;generated+=len(a['generated_ids'])
                answers.flush();tokens.flush();guard()
            try:
                if g['cohort']=='stream34':
                    def event(item):
                        if item['event']['kind']=='query':
                            q=next(q for q in qs if q['id']==item['event']['query_id'])
                            record(q,item['answer'],item['committed_frames_sha256'])
                    replay(source,qs,s,read(run/'snapshots34.json'),event,
                        lambda t:tokens.write(json.dumps(t)+'\n'),read_from='readout_banks')
                else:
                    pending=None
                    for packet in source.packets():
                        if pending is None:pending=packet
                        else:s.append_pair(pending,packet);pending=None
                    if pending is not None:s.snapshot_tail(pending)
                    for q in qs:
                        assert s.pairs==q['pairs']
                        record(q,s.ask(prompt(q),read_from='readout_banks',
                            on_token=lambda t:tokens.write(json.dumps(dict(query_id=q['id'],**t))+'\n')))
                encodes+=s.vision_encodings
                profiles.append(dict(group=g['id'],cohort=g['cohort'],source=source.status(),
                    pairs=s.vision_encodings,bank_status=s.bank.status(),update_seconds=s.bank.update_seconds,
                    vision_timing=s.timing.copy(),shared_pipeline_wall_seconds=time.monotonic()-start))
            finally:s.close();source.close()
            assert not s.bank.banks and s.bank.descriptor is None
            del s;gc.collect();torch.cuda.empty_cache();guard()
            print(dict(groups_complete=number+1,answers=count),flush=True)
    assert count==909 and encodes==3981 and generated+warm_generated<=7296
    save(out/'completed.json',dict(status='complete',answers=count,generated_tokens=generated,
        warmup_generated_tokens=warm_generated,vision_encodings=encodes,warmup_encodings=8,pretrained_loads=1,
        measured_wall_seconds=time.monotonic()-measured,profiles=profiles,measured_probes=303,
        shared_GPU_allocated_peak_bytes=torch.cuda.max_memory_allocated(),model_baseline_bytes=baseline,
        observed_RSS_bytes=env['psutil'].Process().memory_info().rss,shared_costs_not_independent=True,
        no_main_KV=True,no_full_history_archive=True,not_wallclock_1x=True,
        execution_lock_sha256=sha(run/'execution_lock.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();worker(a.run,a.out)
