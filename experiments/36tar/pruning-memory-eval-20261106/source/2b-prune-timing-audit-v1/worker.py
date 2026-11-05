import argparse,os,time,gc,json,hashlib
from pathlib import Path
from collections import deque
from audit_common import *
def main(out):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm required')
    verify();auth=read(PREP/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'runs').resolve())
    qs=read(PREP/'requests.json');contract={x['id']:x for x in read(PREP/'sample_contract.json')}
    refs={(x['task_id'],x['arm'],x['shift']):x['choice'] for x in read(PREP/'reference_choices.json')}
    data=ROOT/'datasets/StreamingBench-eval200-v1/videos'
    for m in read(PREP/'media.json'):
        p=data/m['name'];assert p.resolve().parent==data.resolve() and p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
    env=runtime();t=env['torch'];t.set_num_threads(1)
    assert t.cuda.device_count()==1 and t.cuda.mem_get_info()[0]>=24*2**30
    assert env['psutil'].virtual_memory().available>=12*2**30
    out.mkdir();start=time.monotonic();answers=generated=vision=warmups=regressions=0
    try:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval();model.requires_grad_(False)
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),(288,384))
        def guard():
            if time.monotonic()-start>28*60:raise TimeoutError('28 minute internal stop')
            assert env['psutil'].Process().memory_info().rss<20*2**30 and t.cuda.max_memory_allocated()<32*2**30
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<48*2**20
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream,(out/'pair_audit.jsonl').open('x',encoding='utf-8') as pairs:
            for qi,q in enumerate(qs):
                session=AuditedSession(env,model,builder);recent=deque(maxlen=16)
                def receive(a,b,padding):
                    guard();session.append_pair(a,b,padding=padding)
                    for offset,f in enumerate((a,b)):
                        recent.append(dict(pair_index=session.pairs-1,within_pair=offset,frame_id=f.frame_id,pts_s=f.timestamp,
                            rgb_sha256=hashlib.sha256(f.rgb.tobytes()).hexdigest(),padding=padding))
                    pairs.write(json.dumps(dict(task_id=q['id'],**session.last_pair_audit))+'\n');pairs.flush()
                driver=V0Pairs(data/q['media_name'],receive,config='auto4')
                try:
                    state=driver.advance(q['query_time_s']);expected=contract[q['id']]
                    assert state==expected['state'] and session.pairs==expected['pairs'] and list(recent)==expected['frames']
                    text=question_prompt(rotate(q,0)[0])
                    if not warmups:
                        for mode in MODES:
                            r=session.ask(mode,text,lambda _:None,max_new_tokens=8)
                            warmups+=1;generated+=len(r['generated_ids'])
                    before=session.status();fingerprint=session.bank.selector_digest()
                    for shift in range(4):
                        text=question_prompt(rotate(q,shift)[0]);offset=(qi+shift)%5
                        for mode in MODES[offset:]+MODES[:offset]:
                            guard();r=session.ask(mode,text,lambda _:None,max_new_tokens=8)
                            row=dict(r,task_id=q['id'],arm=mode,shift=shift,source_state=state,choice=choice(r['text'],shift))
                            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                            answers+=1;generated+=len(r['generated_ids'])
                            key=(q['id'],mode,shift)
                            if key in refs:
                                assert row['choice']==refs[key],f'old answer mismatch: {key}'
                                regressions+=1
                    assert session.status()==before and session.bank.selector_digest()==fingerprint
                    vision+=session.visual_calls
                finally:driver.close();session.close()
                del session;gc.collect();t.cuda.empty_cache()
                print(json.dumps(dict(sources=qi+1,answers=answers,regressions=regressions,vision=vision)),flush=True)
        assert (answers,warmups,regressions,vision)==(120,5,72,2565) and generated<=1000
        verify();save(out/'completed.json',dict(status='complete',answers=answers,warmups=warmups,regressions=regressions,
            vision_calls=vision,generated_tokens=generated,wall_seconds=time.monotonic()-start,
            selected_ID_pairs_verified=513*2,pretrained_loads=1,training_steps=0,
            peak_allocated_bytes=t.cuda.max_memory_allocated(),mechanism_only=True,standalone_cost=False))
    except Exception as e:
        save(out/'failed.json',dict(error=str(e),answers=answers,generated_tokens=generated,regressions=regressions));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
