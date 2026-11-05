"""Fixed existing 150-question cross-task panel; no tuning or full feature archive."""
import argparse,os,time,gc,collections
from common import *

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    verify();auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['lock_sha256']==sha(P/'execution_lock.json')
    assert read(P/'CPU_local.json')==read(P/'CPU_cluster.json')
    expected=read(P/'media_verified.json');assert expected['passed']
    assert not out.exists() and out.resolve().is_relative_to((R/'runs').resolve());out.mkdir()
    counts=dict(answers=0,vision=0,warmups=0,numerical=0,generated_tokens=0);start=time.monotonic()
    def guard():
        if time.monotonic()-start>88*60:raise TimeoutError('88-minute stop line')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<64*2**20
    try:
        env=runtime();t=env['torch'];t.set_num_threads(1)
        assert t.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        groups=collections.defaultdict(list)
        for q in read(P/'questions.json'):groups[q['source_id']].append(q)
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as f:
            for index,(source,qs) in enumerate(sorted(groups.items())):
                path=R/qs[0]['media_path'];assert sha(path)==qs[0]['media_sha256']
                session=Session(env,model,builder,arms=('LS_mid108',));d=hashlib.sha256()
                def commit(a,b,padding):
                    guard();pair_digest(d,a,b,padding);session.append_pair(a,b,padding=padding);counts['vision']+=1
                driver=V0Pairs(path,commit,config='auto4')
                try:
                    for q in sorted(qs,key=lambda q:(q['query_time_s'],q['id'])):
                        state=driver.advance(q['query_time_s']);old=expected['prefixes'][q['id']]
                        assert state==old['state'] and d.hexdigest()==old['pixel_sha256'],q['id']
                        h,r=split(session.bank.banks['LS_mid108']);before=session.bank.fingerprint();text=prompt(q)
                        if not counts['warmups']:
                            for arm in ARMS:
                                z=protected_readout.answer(env,model,builder,h if arm!='recent' else [],r,text,isolated=arm=='isolated',limit=1)
                                counts['warmups']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                        responses={};off=counts['answers']//3%3;order=ARMS[off:]+ARMS[:off]
                        for arm in order:
                            guard();t.cuda.synchronize();t.cuda.reset_peak_memory_stats();tick=time.monotonic()
                            if arm=='joint':z=session.ask('LS_mid108',text,lambda _:None,max_new_tokens=16)
                            else:z=compact(protected_readout.answer(env,model,builder,h if arm=='isolated' else [],r,text,isolated=arm=='isolated',limit=16))
                            t.cuda.synchronize();z.update(QA_wall_with_audit_seconds=time.monotonic()-tick,peak_allocated_bytes=t.cuda.max_memory_allocated())
                            responses[arm]=z;counts['answers']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                            f.write(json.dumps(dict(z,id=q['id'],source_id=source,type=q['type'],arm=arm,source_state=state,pixel_sha256=d.hexdigest(),history_nodes=len(h),recent_nodes=len(r),bank_status=session.bank.status()),ensure_ascii=False)+'\n');f.flush()
                        assert responses['joint']['bounded_context_ids_sha256']==responses['isolated']['bounded_context_ids_sha256']
                        assert session.bank.fingerprint()==before
                        if index<8 and q['id']==sorted(qs,key=lambda x:(x['query_time_s'],x['id']))[0]['id']:
                            z=protected_readout.answer(env,model,builder,h,r,text,isolated=False,limit=16)
                            assert z['generated_ids']==responses['joint']['generated_ids'],('segmented_joint',q['id'])
                            counts['numerical']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                finally:driver.close();session.close()
                print(json.dumps(dict(sources=index+1,seconds=time.monotonic()-start,**counts)),flush=True);gc.collect();t.cuda.empty_cache()
        assert counts['answers']==450 and counts['vision']==expected['vision_pairs'] and counts['warmups']==3
        assert counts['generated_tokens']<=7376 and counts['numerical']==8
        verify();save(out/'completed.json',dict(status='complete',**counts,seconds=time.monotonic()-start,model_loads=1,training=0))
    except Exception as e:save(out/'failed.json',dict(error=repr(e),**counts));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
