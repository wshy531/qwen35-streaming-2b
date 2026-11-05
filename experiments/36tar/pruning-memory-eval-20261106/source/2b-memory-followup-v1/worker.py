import argparse,os,time,json,gc,hashlib,collections
from followup import *

def verify():
    for n,h in read(OUT/'execution_lock.json').items():
        p=R/n
        if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
        assert sha(p)==h,n

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    verify();auth=read(OUT/'GPU_authorization.json')
    assert auth['approved'] and auth['lock_sha256']==sha(OUT/'execution_lock.json')
    assert read(OUT/'CPU_local.json')==read(OUT/'CPU_cluster.json')
    assert read(OUT/'media_verified.json')['passed']
    assert not out.exists() and out.resolve().is_relative_to((R/'runs').resolve());out.mkdir()
    m=read(OUT/'manifest.json');expected=read(P/'media_verified.json');bundle=read(HEAD_PATH)
    counts=dict(answers=0,controls=0,warmups=0,vision=0,features=0,generated_tokens=0)
    start=time.monotonic()
    def guard():
        if time.monotonic()-start>108*60:raise TimeoutError('108 minute stop line')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<64*2**20
    def replay(q,z,arm):
        for k,v in m['expected'][q['id']][arm].items():assert z[k]==v,(q['id'],arm,k)
    try:
        env=runtime();t=env['torch'];t.set_num_threads(1)
        assert t.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        groups=collections.defaultdict(list)
        for q in m['questions']:groups[q['source_id']].append(q)
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as f:
            for source,qs in sorted(groups.items()):
                path=R/qs[0]['media_path'];assert sha(path)==qs[0]['media_sha256']
                session=Session(env,model,builder,arms=('LS_mid108','A06_mid108'));d=hashlib.sha256()
                def commit(a,b,padding):
                    guard();pair_digest(d,a,b,padding);session.append_pair(a,b,padding=padding);counts['vision']+=1
                driver=V0Pairs(path,commit,config='auto4')
                try:
                    for q in sorted(qs,key=lambda q:(q['query_time_s'],q['id'])):
                        state=driver.advance(q['query_time_s']);old=expected['prefixes'][q['id']]
                        assert state==old['state'],q['id']
                        if 'pixel_sha256' in old:assert d.hexdigest()==old['pixel_sha256']
                        h,r=split(session.bank.banks['LS_mid108'])
                        if not counts['warmups']:
                            for arm in ('LS_mid108','A06_mid108'):
                                z=session.ask(arm,prompt(q),lambda _:None,max_new_tokens=1)
                                counts['warmups']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                        if q['id'] in m['control_ids']:
                            for arm in ('recent','joint'):
                                guard()
                                z=session.ask('LS_mid108',prompt(q),lambda _:None,max_new_tokens=16) if arm=='joint' else compact(protected_readout.answer(env,model,builder,[],r,prompt(q),isolated=False,limit=16))
                                replay(q,z,arm);counts['controls']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                        order=('API33_LS','A06_joint') if counts['answers']//2%2==0 else ('A06_joint','API33_LS')
                        for arm in order:
                            guard();t.cuda.synchronize();t.cuda.reset_peak_memory_stats();tick=time.monotonic()
                            if arm=='API33_LS':
                                z=answer(env,model,builder,session,q,bundle)
                                counts['features']+=int(z['decision']['head_called'])
                                replay(q,z,'joint' if z['decision']['route']=='full' else 'recent')
                            else:z=session.ask('A06_mid108',prompt(q),lambda _:None,max_new_tokens=16)
                            t.cuda.synchronize()
                            z.update(id=q['id'],source_id=source,type=q['type'],arm=arm,source_state=state,pixel_sha256=d.hexdigest(),
                                QA_wall_with_selection_and_audit_seconds=time.monotonic()-tick,peak_allocated_bytes=t.cuda.max_memory_allocated(),bank_status=session.bank.status())
                            counts['answers']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                            f.write(json.dumps(z,ensure_ascii=False)+'\n');f.flush()
                finally:driver.close();session.close()
                print(json.dumps(dict(seconds=time.monotonic()-start,**counts)),flush=True);gc.collect();t.cuda.empty_cache()
        assert counts['answers']==300 and counts['controls']==32 and counts['warmups']==2 and counts['vision']==25815
        assert counts['generated_tokens']<=5344 and counts['features']<=150
        verify();save(out/'completed.json',dict(**counts,seconds=time.monotonic()-start,model_loads=1,training=0))
    except Exception as e:save(out/'failed.json',dict(error=repr(e),**counts));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
