"""Frozen five-arm readout comparison. No training or paid API calls."""
import argparse,collections,gc,hashlib,json,os,time
from pathlib import Path
from common import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NExT-QA','StreamingBench'],required=True)
    p.add_argument('--minutes',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for n,h in read(P/'execution_lock_v4.json').items():assert sha(R/n)==h,n
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    assert not a.out.exists() and a.out.resolve().is_relative_to((R/'runs').resolve())
    a.out.mkdir();(a.out/'sampled_frames').mkdir();started=time.monotonic()
    answers=calls=tokens=regressions=warmups=png_bytes=0;materials=[];verified=[];saved_images={}
    def guard():
        if time.monotonic()-started>(a.minutes-2)*60:raise TimeoutError('fixed deadline')
        for f in a.out.glob('*.json*'):assert f.stat().st_size<64*2**20
        assert png_bytes<768*2**20
    try:
        env=s.runtime();t=env['torch'];t.set_num_threads(1);assert t.cuda.device_count()==1
        assert env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,
            dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
        builder=s.make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        head=GroupScorer(model.config.vision_config.hidden_size).to('cuda')
        head.load_state_dict(t.load(CHECKPOINT,map_location='cpu',weights_only=True));head.eval().requires_grad_(False)
        qs=[q for q in read(P/'panel.json') if q['dataset']==a.dataset];assert len(qs)==150
        reuse={(x['id'],x['arm']):x for x in read(P/'reuse.json')}
        gates={(x['id'],x['arm']):x for x in read(P/'regression.json')}
        replay_refs={(x['id'],x['arm']):x for x in read(P/'replay_real_reference.json')}
        audits={x['video_id']:x for x in read(s.DATA/'CPU_media_audit.json')['rows']}
        from PIL import Image
        def frame_meta(packet,source_id):
            nonlocal png_bytes
            digest=hashlib.sha256(packet.rgb.tobytes()).hexdigest();key=(source_id,packet.frame_id,digest)
            if key not in saved_images:
                name=hashlib.sha256(repr(key).encode()).hexdigest()[:24]+'.png';dest=a.out/'sampled_frames'/name
                Image.fromarray(packet.rgb).save(dest);png_bytes+=dest.stat().st_size
                saved_images[key]=dict(frame_id=packet.frame_id,pts_s=packet.timestamp,path=dest.relative_to(R).as_posix(),rgb_sha256=digest)
            return saved_images[key]
        def answer_query(q,banks,frames,meta,stream):
            nonlocal answers,tokens,regressions,warmups
            if not warmups:
                for arm in ARMS:
                    guard();z=generate(env,model,builder,list(banks[arm]) if arm!='text_only' else [],q,arm)
                    tokens+=len(z['generated_ids']);warmups+=1
            ordered=ARMS[answers%5:]+ARMS[:answers%5]
            for arm in ordered:
                guard();k=(q['id'],arm);records=list(banks[arm]) if arm!='text_only' else []
                if k in gates:
                    actual=generate(env,model,builder,records,q,arm);tokens+=len(actual['generated_ids']);regressions+=1
                    ok=actual['generated_ids']==gates[k]['generated_ids']
                    with (a.out/'regression.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=q['id'],arm=arm,matched=ok,actual=actual['generated_ids'],expected=gates[k]['generated_ids']))+'\n')
                    assert ok,('old endpoint mismatch',q['id'],arm)
                tick=time.monotonic()
                if k in reuse:
                    result=dict(reuse[k]);result.update(parse(result['text'],len(q['options']),'B'))
                    # Recompute layout identity without another answer generation.
                    _,pos,ids=layout(env,model,builder,records,q,arm)
                    assert ids.shape[1]==result['input_tokens']
                    result.update(input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                        position_ids_sha256=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest(),visual_tokens=sum(len(r[3]) for r in records))
                else:
                    result=generate(env,model,builder,records,q,arm);answers+=1;tokens+=len(result['generated_ids'])
                t.cuda.synchronize()
                if k in replay_refs:
                    old=replay_refs[k]
                    assert result['generated_ids']==old['generated_ids'] and result['input_ids_sha256']==old['input_ids_sha256'] and result['position_ids_sha256']==old['position_ids_sha256'],('real CE replay mismatch',k)
                    with (a.out/'real_CE_replay.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=q['id'],matched=True))+'\n')
                result.update(id=q['id'],source_id=q['source_id'],dataset=a.dataset,type=q['type'],arm=arm,protocol='B',shift=0,
                    query_time_s=q['query_time_s'],visible_pairs=[r[0] for r in records],source_state=meta,seconds=time.monotonic()-tick,
                    token_limit=128,reused=k in reuse)
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            materials.append(dict(material_id=q['id'],source_id=q['source_id'],query_time_s=q['query_time_s'],frames=frames,
                complete_within_method_input=True,evidence_protocol=q['evidence_protocol'],source_state=meta))
            save(a.out/'materials.json',materials)
            print(json.dumps(dict(questions=len(materials),new_answers=answers,regressions=regressions,vision_calls=calls,seconds=time.monotonic()-started)),flush=True)
        # Gates are run before other sources so a mismatch does not consume the full budget.
        gate_sources={q['source_id'] for q in qs if any((q['id'],arm) in gates or (q['id'],arm) in replay_refs for arm in ARMS)}
        sources=sorted({q['source_id'] for q in qs},key=lambda v:(v not in gate_sources,v))
        with t.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for vid in sources:
                guard();vq=sorted((q for q in qs if q['source_id']==vid),key=lambda q:(q['query_time_s'],q['id']));path=R/vq[0]['media_path']
                assert sha(path)==vq[0]['media_sha256'];verified.append(dict(source_id=vid,sha256=vq[0]['media_sha256']))
                if a.dataset=='NExT-QA':
                    pairs=s.decode(env,path,audits[vid],builder);banks={};frames=[]
                    # Export exact pixels already checked by the original decoder contract.
                    from audit_media import canvas
                    wanted={x['frame_index']:x for x in audits[vid]['samples']}
                    with env['av'].open(str(path)) as c:
                        v=c.streams.video[0];v.thread_count=1
                        for i,f in enumerate(c.decode(v)):
                            if i in wanted:
                                rgb=env['np'].asarray(canvas(f));assert hashlib.sha256(rgb.tobytes()).hexdigest()==wanted[i]['rgb_sha256']
                                packet=env['FramePacket'](frame_id=i,timestamp=wanted[i]['pts_s'],rgb=rgb,decoded_count=i+1)
                                frames.append(frame_meta(packet,vid))
                    for arm in ARMS[:-1]:
                        records=[];state=None;selector=s.CoverageScorer()
                        for i,(begin,end,batch) in enumerate(pairs):
                            guard();features,ids,state=encode(model,batch.to('cuda'),arm,head,state,selector);calls+=1
                            records.append((i,begin,end,features,ids))
                        banks[arm]=records
                    for q in vq:answer_query(q,banks,frames,dict(complete_pairs=len(pairs),actual_EOF=True),stream)
                    del banks,pairs,records,features
                else:
                    banks={};rgb_pairs=collections.deque(maxlen=9)
                    index=0
                    def commit(first,second,padding):
                        nonlocal index
                        guard();rgb_pairs.append((index,first,second,padding));index+=1
                    driver=CausalPairs(path,commit)
                    try:
                        for q in vq:
                            state=driver.advance(q['query_time_s']);assert rgb_pairs
                            # Selection at pair t depends only on the unpruned layer-11
                            # groups at t-1. One preceding pair restores this state exactly.
                            begin_index=max(0,index-8);banks={arm:[] for arm in ARMS[:-1]}
                            selectors={arm:s.CoverageScorer() for arm in ARMS[:-1]};states={arm:None for arm in ARMS[:-1]}
                            for pi,first,second,padding in rgb_pairs:
                                guard();batch,_=builder.pair(first,second,0,0,synthetic_padding=padding);batch=batch.to('cuda')
                                for arm in ARMS[:-1]:
                                    features,ids,states[arm]=encode(model,batch,arm,head,states[arm],selectors[arm]);calls+=1
                                    if pi>=begin_index:banks[arm].append((pi,first.timestamp,second.timestamp,features,ids))
                            assert len(banks['full'])==min(index,8)
                            assert all(r[2]<=q['query_time_s']+1e-6 for r in banks['full'])
                            frames=[];seen=set()
                            for pi,first,second,padding in rgb_pairs:
                                if pi<begin_index:continue
                                for packet in (first,second):
                                    if packet.frame_id not in seen:frames.append(frame_meta(packet,vid));seen.add(packet.frame_id)
                            assert all(f['pts_s']<=q['query_time_s']+1e-6 for f in frames)
                            answer_query(q,banks,frames,state,stream)
                    finally:driver.close()
                    del banks,rgb_pairs,selectors,states
                gc.collect();t.cuda.empty_cache()
        spec=read(P/'selection.json')['datasets'][a.dataset]
        assert answers==660 and regressions==9 and warmups==5 and len(materials)==150
        assert len((a.out/'real_CE_replay.jsonl').read_text().splitlines())==len(replay_refs)==7
        assert calls<=(5400 if a.dataset=='StreamingBench' else spec['vision_call_upper_bound']) and tokens<=674*128
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        save(a.out/'completed.json',dict(status='complete_generation_only',dataset=a.dataset,questions=150,new_answers=answers,reused_answers=90,
            regression_answers=regressions,warmups=warmups,vision_calls=calls,generated_tokens=tokens,seconds=time.monotonic()-started,
            model_loads=1,training=0,API=0,base_unchanged=True,checkpoint_sha256=CHECKPOINT_SHA,checked_media=verified,png_bytes=png_bytes,
            peak_allocated_bytes=t.cuda.max_memory_allocated(),cost_scope='causal CPU sampling; on-query recent8 plus predecessor replay; quality only, not continuous encoding speed'))
    except Exception as e:
        save(a.out/'failed.json',dict(error=repr(e),new_answers=answers,vision_calls=calls,regressions=regressions,seconds=time.monotonic()-started));raise
if __name__=='__main__':main()
