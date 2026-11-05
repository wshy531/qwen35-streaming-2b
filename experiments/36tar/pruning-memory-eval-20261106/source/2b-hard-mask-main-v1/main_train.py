"""One bounded repair: two one-epoch heads, unchanged base and frozen QA protocol."""
import argparse, gc, importlib.util, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-learned-prune-main-v1'))
from protocol import runtime,make_builder,decode,read,sha,DATA,PREP as OLD_PREP,rotate,context,question_text,encode_pair,CoverageScorer
sys.path.insert(0,str(ROOT/'experiments/2b-hard-mask-prune-v1'))
from hard_prune import torch,GroupScorer,encode
from objective import last_logits,option_ids,loss_parts
spec=importlib.util.spec_from_file_location('original_prune_worker',ROOT/'experiments/2b-learned-prune-main-v1/worker.py')
old_worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_worker)
PREP=ROOT/'runs/hard_mask_main_prepare_20261102_01'
ARMS=('mask_initial','mask_ce','mask_ce_kl')
BASELINES=('full','fixed12','heuristic11')

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def records_for(model,pairs,scorer,mode):
    records=[];state=None
    for i,(begin,end,batch) in enumerate(pairs):
        batch=batch.to(model.get_input_embeddings().weight.device)
        feat,ids,_,state=encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,
                              scorer,previous=state,mode=mode)
        records.append((i,begin,end,feat,ids))
    return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for name,digest in read(PREP/'execution_lock.json').items():assert sha(ROOT/name)==digest,name
    assert not a.out.exists() and a.out.resolve().is_relative_to((ROOT/'runs').resolve())
    spec=read(PREP/'protocol.json');a.out.mkdir();started=time.monotonic();updates=answers=vision=teacher_forwards=0
    def guard():
        if time.monotonic()-started>178*60:raise TimeoutError('bounded repair deadline')
        assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<128*2**20
    try:
        env=runtime();torch.set_num_threads(1)
        assert torch.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
            local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        train=read(OLD_PREP/'train.json');evaluation=read(OLD_PREP/'eval.json')
        sources={s['video_id']:s for s in read(OLD_PREP/'sources.json')}
        audits={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
        verified=set()
        def load_pairs(vid):
            source=sources[vid];path=ROOT/'datasets/learned-prune-nextqa-v1'/source['mirror_path']
            if vid not in verified:assert sha(path)==source['mirror_lfs_sha256'];verified.add(vid)
            return decode(env,path,audits[vid],builder)
        scorer=GroupScorer(model.config.vision_config.hidden_size).to('cuda')
        initial=torch.load(ROOT/'runs/learned-prune-main-220024/scorer_untrained.pt',map_location='cpu',weights_only=True)
        scorer.load_state_dict(initial);options=option_ids(builder,torch.device('cuda'))
        # Reproduce a preselected source-separated subset before reusing old baseline answers.
        references={ (r['id'],r['arm'],r['shift']):r for r in read(PREP/'reference_subset.json') }
        gates=read(PREP/'reference_questions.json');regressions=[]
        with torch.no_grad(),env['fp32_reference']():
            for q in gates:
                guard();pairs=load_pairs(q['video_id'])
                for arm in BASELINES:
                    records=old_worker.records_for(model,pairs,arm,None);vision+=len(pairs)
                    for shift in (0,2):
                        result=old_worker.answer(env,model,builder,records,rotate(q,shift));answers+=1
                        expected=references[q['id'],arm,shift]
                        assert result['input_ids_sha256']==expected['input_ids_sha256'] and result['generated_ids']==expected['generated_ids'],(q['id'],arm,shift)
                        regressions.append(dict(id=q['id'],arm=arm,shift=shift,generated_ids=result['generated_ids'],matched=True))
                del pairs,records;gc.collect()
        save(a.out/'baseline_regression.json',regressions)
        # Teacher is frozen. Reuse its visual encoding within one source, never across sources.
        teachers={};teacher_start=time.monotonic()
        with torch.no_grad(),env['fp32_reference']():
            for vid in sorted({q['video_id'] for q in train}):
                guard();assert sources[vid]['role']=='train';pairs=load_pairs(vid)
                records=old_worker.records_for(model,pairs,'full',None);vision+=len(pairs)
                for q in [q for q in train if q['video_id']==vid]:
                    guard();rq=rotate(q,q['shift']);logits,_=last_logits(model,builder,records,rq);teacher_forwards+=1
                    teachers[q['id']]=logits[:,options].detach().cpu()
                del records,pairs,logits;gc.collect()
        torch.save(teachers,a.out/'teacher_options.pt')
        save(a.out/'teacher_summary.json',dict(questions=len(teachers),correct=sum(int(teachers[q['id']].argmax())==rotate(q,q['shift'])['answer_index'] for q in train),
              seconds=time.monotonic()-teacher_start,source='frozen identical 2B full path',
              correctness_criterion='argmax among A-E logits, not a generated-answer evaluation',evaluation_filtered=False))
        endpoints={'mask_initial':initial};train_times={}
        for arm in ('mask_ce','mask_ce_kl'):
            scorer.load_state_dict(initial);opt=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
            tick_arm=time.monotonic();arm_updates=0
            with (a.out/f'{arm}_train.jsonl').open('x',encoding='utf-8') as stream:
                for q in train:
                    guard();assert q['role']=='train';pairs=load_pairs(q['video_id']);rq=rotate(q,q['shift'])
                    opt.zero_grad(set_to_none=True);tick=time.monotonic()
                    with env['fp32_reference']():
                        records=records_for(model,pairs,scorer,'mask');vision+=len(pairs)
                        logits,tokens=last_logits(model,builder,records,rq)
                        loss,ce,kl,active=loss_parts(logits,options,rq['answer_index'],teachers[q['id']].to('cuda') if arm=='mask_ce_kl' else None)
                        assert torch.isfinite(loss);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.)
                    assert torch.isfinite(grad) and all(p.grad is None or torch.isfinite(p.grad).all() for p in scorer.parameters())
                    assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
                    opt.step();torch.cuda.synchronize();updates+=1;arm_updates+=1
                    stream.write(json.dumps(dict(id=q['id'],shift=q['shift'],loss=float(loss.detach()),ce=float(ce.detach()),kl=float(kl.detach()),
                         kl_active=active,grad_norm=float(grad),tokens=tokens,seconds=time.monotonic()-tick))+'\n');stream.flush()
                    if arm_updates in (310,619,928,1237):
                        torch.save(scorer.state_dict(),a.out/f'{arm}_step{arm_updates}.pt')
                    if arm_updates%100==0:print(json.dumps(dict(stage='train',arm=arm,updates=arm_updates,seconds=time.monotonic()-started)),flush=True)
                    del records,pairs,logits,loss,ce,kl;gc.collect()
            assert arm_updates==1237
            endpoints[arm]={n:p.detach().cpu().clone() for n,p in scorer.state_dict().items()}
            assert any(not torch.equal(initial[n],endpoints[arm][n]) for n in initial)
            train_times[arm]=time.monotonic()-tick_arm;del opt;torch.cuda.empty_cache()
        with torch.no_grad(),env['fp32_reference'](),(a.out/'training_probes.jsonl').open('x',encoding='utf-8') as stream:
            for q in read(PREP/'training_probes.json'):
                guard();pairs=load_pairs(q['video_id']);rq=rotate(q,q['shift'])
                for arm in ARMS:
                    scorer.load_state_dict(endpoints[arm]);records=records_for(model,pairs,scorer,'gather');vision+=len(pairs)
                    logits,_=last_logits(model,builder,records,rq)
                    _,ce,kl,active=loss_parts(logits,options,rq['answer_index'],teachers[q['id']].to('cuda'))
                    stream.write(json.dumps(dict(id=q['id'],arm=arm,ce=float(ce),kl=float(kl),kl_active=active,
                        option_correct=int(logits[:,options].argmax())==rq['answer_index'],
                        first_token_correct=int(logits.argmax())==int(options[rq['answer_index']])))+'\n');stream.flush()
                del pairs,records,logits;gc.collect()
        generated=0
        with torch.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for vi,vid in enumerate(sorted({q['video_id'] for q in evaluation})):
                guard();assert sources[vid]['role']!='train';pairs=load_pairs(vid)
                for arm in ARMS[vi%3:]+ARMS[:vi%3]:
                    scorer.load_state_dict(endpoints[arm]);records=records_for(model,pairs,scorer,'gather');vision+=len(pairs)
                    for q in [q for q in evaluation if q['video_id']==vid]:
                        for shift in (0,2):
                            guard();result=old_worker.answer(env,model,builder,records,rotate(q,shift));answers+=1;generated+=len(result['generated_ids'])
                            result.update(id=q['id'],video_id=vid,role=q['role'],type=q['type'],arm=arm,shift=shift,
                                original_selected_index=None if result['selected_index'] is None else (result['selected_index']+shift)%5)
                            stream.write(json.dumps(result)+'\n');stream.flush()
                    del records;gc.collect()
                del pairs;torch.cuda.empty_cache()
                print(json.dumps(dict(stage='evaluation',sources=vi+1,answers=answers,seconds=time.monotonic()-started)),flush=True)
        total_generated=generated+sum(len(r['generated_ids']) for r in regressions)
        assert updates==2474 and answers==spec['total_answers'] and teacher_forwards==1237 and vision==spec['vision_calls']
        assert total_generated<=spec['max_generated_tokens']
        save(a.out/'completed.json',dict(status='complete',updates=updates,answers=answers,teacher_forwards=teacher_forwards,vision_calls=vision,
            generated_tokens=total_generated,model_loads=1,seconds=time.monotonic()-started,training_seconds=train_times,
            base_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),shared_decode_not_standalone_cost=True))
    except Exception as e:
        save(a.out/'failed.json',dict(error=repr(e),updates=updates,answers=answers,teacher_forwards=teacher_forwards,vision_calls=vision));raise

if __name__=='__main__':main()
