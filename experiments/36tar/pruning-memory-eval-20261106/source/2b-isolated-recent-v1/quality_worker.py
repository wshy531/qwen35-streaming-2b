"""One shared-vision causal replay; no full feature archive or observer language KV."""
import argparse,os,time,gc,importlib.util
from experiment import *

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    verify();auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['lock_sha256']==sha(P/'execution_lock.json')
    assert read(P/'CPU_local.json')==read(P/'CPU_cluster.json')
    assert not out.exists() and out.resolve().is_relative_to((R/'runs').resolve());out.mkdir()
    plan=read(P/'protocol.json');start=time.monotonic();counts=dict(answers=0,vision=0,warmups=0,numerical=0,controls=0,generated_tokens=0)
    def guard():
        if time.monotonic()-start>88*60:raise TimeoutError('fixed 88-minute deadline; preserve partial run')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<96*2**20
    try:
        env=runtime();t=env['torch'];t.set_num_threads(1)
        assert t.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        refs={q['question_id']:q for q in read(P/'baseline.json')}
        def append(session,a,b,**kw):
            guard();session.append_pair(a,b,**kw);counts['vision']+=1
        def ask_all(session,key,text,extra,stream,reference=None,numeric=False):
            guard();h,r=split(session.bank.banks['LS_mid108']);before=session.bank.fingerprint()
            if not counts['warmups']:
                for arm in ARMS:
                    z=protected_readout.answer(env,model,builder,h if arm!='recent' else [],r,text,isolated=arm=='isolated',limit=1)
                    counts['warmups']+=1;counts['generated_tokens']+=len(z['generated_ids'])
            # Rotate execution order, not question order, to limit timing bias.
            off=counts['answers']//3%3;order=ARMS[off:]+ARMS[:off]
            responses={}
            for arm in order:
                guard();t.cuda.synchronize();t.cuda.reset_peak_memory_stats();tick=time.monotonic()
                if arm=='joint':z=session.ask('LS_mid108',text,lambda _:None,max_new_tokens=16)
                else:z=compact(protected_readout.answer(env,model,builder,h if arm=='isolated' else [],r,text,isolated=arm=='isolated',limit=16))
                t.cuda.synchronize();z.update(QA_wall_with_audit_seconds=time.monotonic()-tick,peak_allocated_bytes=t.cuda.max_memory_allocated())
                counts['answers']+=1;counts['generated_tokens']+=len(z['generated_ids']);responses[arm]=z
                row=dict(z,arm=arm,id=key,**extra,bank_status=session.bank.status(),history_nodes=len(h),recent_nodes=len(r))
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            assert session.bank.fingerprint()==before
            assert responses['joint']['bounded_context_ids_sha256']==responses['isolated']['bounded_context_ids_sha256']
            if reference:
                for k in ('generated_ids','text','bounded_context_ids_sha256','visual_tokens'):assert responses['joint'][k]==reference[k],(key,k)
                counts['controls']+=1
            if numeric:
                # Check segmented ordinary readout against the legacy joint implementation.
                z=protected_readout.answer(env,model,builder,h,r,text,isolated=False,limit=16)
                assert z['generated_ids']==responses['joint']['generated_ids'],('segmented_joint_answer',key)
                saved=model.model.rope_deltas;branch=env['HybridSequence'](model,max_tokens=4096)
                try:
                    with t.inference_mode(),env['fp32_reference']():
                        emb,pos,ids,meta=context(model,builder,h+r,text,max_new_tokens=16)
                        for lo in range(0,emb.shape[1],512):
                            hi=min(lo+512,emb.shape[1]);logits=branch.append(inputs_embeds=emb[:,lo:hi],position_ids=pos[:,:,lo:hi],logits_mode='last' if hi==emb.shape[1] else 'none')
                        x=logits.cpu();y=z['first_logits'];delta=float((x-y).abs().max());rel=float((x-y).norm()/x.norm().clamp_min(1e-12))
                        assert rel<=1e-4 and x.argmax(-1).item()==y.argmax(-1).item(),('numeric',key,delta,rel)
                finally:branch.cache.reset();model.model.rope_deltas=saved
                counts['numerical']+=1;counts['generated_tokens']+=len(z['generated_ids'])
                with (out/'numerical.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=key,max_abs=delta,relative_l2=rel,answer_exact=True))+'\n')
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            # Familiar current/old diagnostic first; it does not determine main-panel inclusion.
            module=R/'experiments/2b-midprune-memory-v1/pilot.py'
            sys.path.insert(0,str(module.parent))
            spec=importlib.util.spec_from_file_location('isolated_legacy_pairs',module);legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
            pq=read(P/'paired_questions.json')
            for source in read(P/'paired_sources.json'):
                path=Path(plan['paired_media_root'])/source['name'];assert sha(path)==source['media_sha256']
                session=Session(env,model,builder,arms=('LS_mid108',));delivered=set()
                try:
                    for a,b in legacy.exact_pairs(env,path,source['sampled_trace']):
                        append(session,a,b)
                        for q in pq:
                            if q['source_name']!=source['name'] or q['query_time_s']!=b.timestamp:continue
                            assert q['id'] not in delivered;delivered.add(q['id'])
                            for shift in plan['paired_rotations']:
                                rotated,_=rotate(q,shift)
                                ask_all(session,q['id'],question_prompt(rotated),dict(panel='paired',shift=shift,role=q['role'],case_id=q['case_id'],stage=q['stage'],source=source['name']),stream)
                finally:session.close()
                assert delivered=={q['id'] for q in pq if q['source_name']==source['name']}
                print(json.dumps(dict(panel='paired',source=source['name'],**counts)),flush=True);gc.collect();t.cuda.empty_cache()
            qs=read(P/'requests.json');expected=read(P/'expected_mid_prefixes.json')
            for si,v in enumerate(sorted(read(P/'videos.json'),key=lambda v:v['video_id'])):
                path=DATA/v['relative_path'];assert path.resolve().is_relative_to(DATA.resolve()) and sha(path)==v['sha256']
                session=Session(env,model,builder,arms=('LS_mid108',));driver=PairDriver(CausalSource(path),lambda a,b,**kw:append(session,a,b,**kw))
                try:
                    for q in sorted((q for q in qs if q['video_id']==v['video_id']),key=lambda q:(q['question_time'],q['question_id'])):
                        prefix=driver.advance(q['question_time']);old=expected[q['question_id']]
                        for k in ('real_frames','pairs','pending_real_frames','last_committed_pts','submitted_prefix_sha256','true_EOF_tail_padded'):assert prefix[k]==old[k],(q['question_id'],k)
                        ask_all(session,q['question_id'],prompt(q),dict(panel='official',shift=0,category=q['question_category'],source=v['video_id'],query_prefix=prefix),stream,reference=refs[q['question_id']],numeric=q['question_id'] in plan['numeric_ids'])
                finally:driver.close();session.close()
                print(json.dumps(dict(panel='official',sources=si+1,**counts)),flush=True);gc.collect();t.cuda.empty_cache()
        assert counts['answers']==1704 and counts['controls']==478 and counts['numerical']==8 and counts['warmups']==3
        assert counts['vision']==plan['vision_calls'] and counts['generated_tokens']<=plan['max_generated_tokens']
        verify();save(out/'completed.json',dict(status='complete',**counts,seconds=time.monotonic()-start,model_loads=1,training=0,shared_quality_not_standalone_cost=True))
    except Exception as e:save(out/'failed.json',dict(error=repr(e),**counts));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)

