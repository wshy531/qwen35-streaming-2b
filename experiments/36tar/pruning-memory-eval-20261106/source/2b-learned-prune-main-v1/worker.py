"""One fixed epoch and six-arm held-out QA. Explicit Slurm approval required."""
import argparse,gc,json,os,re,time
from pathlib import Path
from protocol import *

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def records_for(model,pairs,arm,scorer):
    records=[];state=None;heuristic=CoverageScorer()
    for index,(begin,end,batch) in enumerate(pairs):
        batch=batch.to(model.get_input_embeddings().weight.device)
        features,ids,state=encode_pair(model,batch,arm,scorer,state,heuristic)
        records.append((index,begin,end,features,ids))
    return records

def answer(env,model,builder,records,q,limit=8):
    emb,pos,ids=context(model,builder,records,question_text(q))
    branch=env['HybridSequence'](model,max_tokens=4096,session_id='P1-fresh-QA')
    saved=model.model.rope_deltas
    try:
        for begin in range(0,emb.shape[1],512):
            end=min(begin+512,emb.shape[1])
            logits=branch.append(inputs_embeds=emb[:,begin:end],position_ids=pos[:,:,begin:end],
                                 logits_mode='last' if end==emb.shape[1] else 'none')
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
        generated,reason=env['greedy_decode'](branch,logits,limit,stops)
    finally:
        branch.cache.reset();model.model.rope_deltas=saved
    text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True)
    letters=set(re.findall(r'\b([A-E])\b',text))
    selected=ord(next(iter(letters)))-65 if len(letters)==1 else None
    return dict(text=text,generated_ids=generated,reason=reason,selected_index=selected,
                correct=selected==q['answer_index'],format_only_letter=text.strip() in list('ABCDE'),
                input_tokens=ids.shape[1],input_ids_sha256=__import__('hashlib').sha256(ids.numpy().tobytes()).hexdigest())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for name,digest in read(PREP/'execution_lock.json').items():assert sha(ROOT/name)==digest,name
    authorization=read(PREP/'GPU_authorization.json');assert authorization['approved'] and authorization['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    assert not a.out.exists() and a.out.resolve().is_relative_to((ROOT/'runs').resolve())
    env=runtime();t=env['torch'];t.set_num_threads(1)
    assert t.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
    spec=read(PREP/'protocol.json');sources={s['video_id']:s for s in read(PREP/'sources.json')}
    train=read(PREP/'train.json');evaluation=read(PREP/'eval.json')
    audit={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
    a.out.mkdir();started=time.monotonic();updates=answers=generated=vision=0
    def guard():
        if time.monotonic()-started>spec['deadline_minutes']*60:raise TimeoutError('fixed deadline; preserve partial evidence')
        assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<128*2**20
    try:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        model.eval().requires_grad_(False)
        versions={n:p._version for n,p in model.named_parameters()}
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        t.manual_seed(SEED);scorer=GroupScorer(model.config.vision_config.hidden_size).to('cuda')
        initial={n:p.detach().cpu().clone() for n,p in scorer.state_dict().items()}
        t.save(initial,a.out/'scorer_untrained.pt')
        optimizer=t.optim.AdamW(scorer.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
        t.cuda.reset_peak_memory_stats()
        verified=set()
        def load_pairs(vid):
            s=sources[vid];path=ROOT/'datasets/learned-prune-nextqa-v1'/s['mirror_path']
            if vid not in verified:assert sha(path)==s['mirror_lfs_sha256'];verified.add(vid)
            return decode(env,path,audit[vid],builder)
        with (a.out/'train.jsonl').open('x',encoding='utf-8') as stream:
            for q in train:
                guard();assert q['role']=='train';pairs=load_pairs(q['video_id'])
                optimizer.zero_grad(set_to_none=True);tick=time.monotonic()
                with env['fp32_reference']():
                    records=records_for(model,pairs,'trained',scorer);vision+=len(pairs)
                    rq=rotate(q,q['shift']);loss,tokens=task_loss(model,builder,records,question_text(rq),chr(65+rq['answer_index']))
                    assert t.isfinite(loss);loss.backward()
                grad=t.nn.utils.clip_grad_norm_(scorer.parameters(),spec['clip_gradient_norm'])
                assert t.isfinite(grad) and all(p.grad is None or t.isfinite(p.grad).all() for p in scorer.parameters())
                assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
                optimizer.step();t.cuda.synchronize();updates+=1
                stream.write(json.dumps(dict(id=q['id'],shift=q['shift'],loss=float(loss.detach()),grad_norm=float(grad),
                    input_tokens=tokens,seconds=time.monotonic()-tick))+'\n');stream.flush()
                del records,pairs,loss;gc.collect()
                if updates%100==0:print(json.dumps(dict(stage='train',updates=updates,seconds=time.monotonic()-started)),flush=True)
        assert updates==1237
        t.save(scorer.state_dict(),a.out/'scorer_final.pt')
        trained={n:p.detach().cpu().clone() for n,p in scorer.state_dict().items()}
        assert any(not t.equal(trained[n],initial[n]) for n in trained)
        del optimizer;t.cuda.empty_cache();training_seconds=time.monotonic()-started
        with t.inference_mode(),(a.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            vids=sorted({q['video_id'] for q in evaluation})
            for vi,vid in enumerate(vids):
                assert sources[vid]['role']!='train';pairs=load_pairs(vid)
                questions=[q for q in evaluation if q['video_id']==vid]
                order=ARMS[vi%6:]+ARMS[:vi%6]
                for arm in order:
                    guard()
                    if arm in ('untrained','trained'):scorer.load_state_dict(initial if arm=='untrained' else trained)
                    with env['fp32_reference']():
                        records=records_for(model,pairs,arm,scorer);vision+=len(pairs)
                        for q in questions:
                            for shift in spec['eval_rotations']:
                                guard();tick=time.monotonic();result=answer(env,model,builder,records,rotate(q,shift));t.cuda.synchronize()
                                generated+=len(result['generated_ids']);answers+=1
                                result.update(id=q['id'],video_id=vid,role=q['role'],type=q['type'],arm=arm,shift=shift,
                                    original_selected_index=None if result['selected_index'] is None else (result['selected_index']+shift)%5,
                                    QA_seconds=time.monotonic()-tick)
                                stream.write(json.dumps(result)+'\n');stream.flush()
                    del records;gc.collect()
                del pairs;t.cuda.empty_cache()
                print(json.dumps(dict(stage='evaluation',sources=vi+1,answers=answers,seconds=time.monotonic()-started)),flush=True)
        assert answers==spec['eval_answers'] and vision==spec['training_vision_calls']+spec['eval_vision_calls']
        assert generated<=spec['max_generated_tokens']
        save(a.out/'completed.json',dict(status='complete',updates=updates,answers=answers,vision_calls=vision,
            generated_tokens=generated,seconds=time.monotonic()-started,training_seconds=training_seconds,
            peak_allocated_bytes=t.cuda.max_memory_allocated(),base_unchanged=True,
            model_loads=1,shared_decode_quality_run_not_standalone_cost=True))
    except Exception as e:
        save(a.out/'failed.json',dict(error=repr(e),updates=updates,answers=answers,vision_calls=vision,generated_tokens=generated));raise

if __name__=='__main__':main()
