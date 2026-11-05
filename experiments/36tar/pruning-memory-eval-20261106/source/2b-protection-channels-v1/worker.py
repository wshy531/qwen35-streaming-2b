import argparse,os,time,gc,importlib.util,json
from channel_readout import *
from experiment import runtime,make_builder,Session,split,read,save,sha,question_prompt,rotate
P=R/'runs/protection_channels_prepare_20261103_01';OLD=R/'runs/isolated_recent_prepare_20261103_01'
def verify():
    for n,h in read(P/'execution_lock.json').items():
        p=R/n
        if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
        assert sha(p)==h,n
def main(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    verify();auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['lock_sha256']==sha(P/'execution_lock.json')
    assert read(P/'CPU_local.json')==read(P/'CPU_cluster.json')
    assert not out.exists() and out.resolve().is_relative_to((R/'runs').resolve());out.mkdir()
    start=time.monotonic();counts=dict(answers=0,vision=0,warmups=0,controls=0,generated_tokens=0)
    def guard():
        if time.monotonic()-start>18*60:raise TimeoutError('18-minute stop line')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<32*2**20
    try:
        env=runtime();t=env['torch'];t.set_num_threads(1)
        assert t.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        refs={(r['id'],r['shift'],r['arm']):r for r in read(P/'baseline.json')};assert len(refs)==180
        module=R/'experiments/2b-midprune-memory-v1/pilot.py';sys.path.insert(0,str(module.parent))
        spec=importlib.util.spec_from_file_location('channels_legacy_pairs',module);legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
        qs=read(OLD/'paired_questions.json');plan=read(OLD/'protocol.json')
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as f:
            for source in read(OLD/'paired_sources.json'):
                path=Path(plan['paired_media_root'])/source['name'];assert sha(path)==source['media_sha256']
                session=Session(env,model,builder,arms=('LS_mid108',));delivered=set()
                try:
                    for a,b in legacy.exact_pairs(env,path,source['sampled_trace']):
                        guard();session.append_pair(a,b);counts['vision']+=1
                        for q in qs:
                            if q['source_name']!=source['name'] or q['query_time_s']!=b.timestamp:continue
                            assert q['id'] not in delivered;delivered.add(q['id']);h,r=split(session.bank.banks['LS_mid108'])
                            for shift in range(5):
                                guard();text=question_prompt(rotate(q,shift)[0]);before=session.bank.fingerprint()
                                if not counts['warmups']:
                                    z=answer_four(env,model,builder,h,r,text,limit=1);counts['warmups']=4
                                    counts['generated_tokens']+=sum(len(x['generated_ids']) for x in z.values())
                                rows=answer_four(env,model,builder,h,r,text,limit=16)
                                assert session.bank.fingerprint()==before
                                for arm,z in rows.items():
                                    counts['answers']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                                    f.write(json.dumps(dict(z,id=q['id'],shift=shift,arm=arm,role=q['role'],stage=q['stage'],case_id=q['case_id'],source=source['name'],bank_state_unchanged=True),ensure_ascii=False)+'\n');f.flush()
                                    if arm in ('joint','isolated'):
                                        ref=refs[q['id'],shift,arm]
                                        for key in ('generated_ids','text','bounded_context_ids_sha256','visual_tokens'):assert z[key]==ref[key],(q['id'],shift,arm,key)
                                        counts['controls']+=1
                finally:session.close()
                assert delivered=={q['id'] for q in qs if q['source_name']==source['name']}
                print(json.dumps(dict(source=source['name'],seconds=time.monotonic()-start,**counts)),flush=True);gc.collect();t.cuda.empty_cache()
        assert counts['answers']==360 and counts['controls']==180 and counts['vision']==429 and counts['warmups']==4
        assert counts['generated_tokens']<=5824
        verify();save(out/'completed.json',dict(status='complete',**counts,seconds=time.monotonic()-start,model_loads=1,training=0,cost_caveat='shared preparation; not a performance comparison'))
    except Exception as e:save(out/'failed.json',dict(error=repr(e),**counts));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
