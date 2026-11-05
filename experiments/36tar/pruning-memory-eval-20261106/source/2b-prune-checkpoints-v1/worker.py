"""Evaluate saved CE scorers only; preserve old endpoints before examining intermediate checkpoints."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-hard-mask-main-v1'))
from main_train import runtime,make_builder,decode,read,sha,DATA,records_for,old_worker,torch,GroupScorer,last_logits,option_ids,loss_parts,context,question_text
P=ROOT/'runs/prune_checkpoints_prepare_20261102_01'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load_heads(model):
    scorer=GroupScorer(model.config.vision_config.hidden_size).to(model.get_input_embeddings().weight.device).eval()
    heads={}
    for arm,n in read(P/'protocol.json')['checkpoints'].items():
        state=torch.load(ROOT/n,map_location='cpu',weights_only=True)
        assert sum(x.numel() for x in state.values())==69697 and all(torch.isfinite(x).all() for x in state.values())
        scorer.load_state_dict(state);heads[arm]=state
    return scorer,heads
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for n,h in read(P/'execution_lock.json').items():assert sha(ROOT/n)==h,n
    auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['execution_lock_sha256']==sha(P/'execution_lock.json')
    for name in ('CPU_local.json','CPU_cluster.json','CPU_media_cluster.json'):assert read(P/name)['passed']
    assert not a.out.exists() and a.out.resolve().is_relative_to((ROOT/'runs').resolve())
    a.out.mkdir();start=time.monotonic();answers=vision=loss_forwards=generated=replays=0
    spec=read(P/'protocol.json');questions=read(P/'questions.json')
    refs={(r['id'],r['arm']):r for r in read(P/'reference.json')}
    sources={s['video_id']:s for s in read(P/'sources.json')}
    audits={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
    def guard():
        if time.monotonic()-start>58*60:raise TimeoutError('fixed 58-minute stop; no automatic retry')
        assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<64*2**20
    try:
        env=runtime();torch.set_num_threads(1)
        assert torch.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        scorer,heads=load_heads(model);scorer.requires_grad_(False);options=option_ids(builder,torch.device('cuda'))
        with torch.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as out,(a.out/'selections.jsonl').open('x',encoding='utf-8') as sel:
            for phase,arms in enumerate(spec['phases']):
                if phase==1:assert replays==690
                for vi,vid in enumerate(sorted(sources)):
                    guard();s=sources[vid];assert s['role']=='development'
                    path=ROOT/'datasets/learned-prune-nextqa-v1'/s['mirror_path']
                    assert sha(path)==s['mirror_lfs_sha256']
                    pairs=decode(env,path,audits[vid],builder)
                    for arm in arms:
                        scorer.load_state_dict(heads[arm]);records=records_for(model,pairs,scorer,'gather');vision+=len(pairs)
                        selected=[dict(pair=r[0],begin=r[1],end=r[2],ids=r[4].detach().cpu().tolist()) for r in records]
                        sel.write(json.dumps(dict(video_id=vid,arm=arm,pairs=selected))+'\n');sel.flush()
                        for q in (q for q in questions if q['video_id']==vid):
                            guard();result=old_worker.answer(env,model,builder,records,q);answers+=1;generated+=len(result['generated_ids'])
                            replay=None
                            if phase==0:
                                ref=refs[q['id'],'mask_initial' if arm=='step0' else 'mask_ce']
                                replay=all(result[k]==ref[k] for k in ('generated_ids','input_ids_sha256','input_tokens','selected_index','correct'))
                                if not replay:
                                    save(a.out/'endpoint_mismatch.json',dict(id=q['id'],arm=arm,actual=result,expected=ref))
                                    raise AssertionError('old endpoint mismatch; do not run intermediate checkpoints')
                                replays+=1
                            saved=model.model.rope_deltas
                            try:
                                logits,tokens=last_logits(model,builder,records,q);loss_forwards+=1
                                _,ce,_,_=loss_parts(logits,options,q['answer_index'])
                                assert torch.isfinite(logits).all() and torch.isfinite(ce) and tokens==result['input_tokens']
                                metrics=dict(ce=float(ce),option_logits=logits[:,options].detach().cpu().flatten().tolist(),
                                    option_argmax_correct=int(logits[:,options].argmax())==q['answer_index'],
                                    first_token_correct=int(logits.argmax())==int(options[q['answer_index']]))
                            finally:model.model.rope_deltas=saved
                            result.update(id=q['id'],video_id=vid,type=q['type'],role='development',arm=arm,shift=0,
                                original_selected_index=result['selected_index'],endpoint_replay=replay,**metrics)
                            out.write(json.dumps(result)+'\n');out.flush();del logits,ce
                        del records;gc.collect()
                    del pairs;torch.cuda.empty_cache()
                    print(json.dumps(dict(phase=phase,sources=vi+1,answers=answers,replays=replays,seconds=time.monotonic()-start)),flush=True)
                if phase==0:save(a.out/'endpoint_gate.json',dict(passed=True,replayed=replays))
        assert answers==spec['answers'] and vision==spec['visual_calls'] and loss_forwards==spec['validation_loss_forwards'] and replays==690
        assert generated<=spec['max_generated_tokens']
        assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
        save(a.out/'completed.json',dict(status='complete',answers=answers,visual_calls=vision,validation_loss_forwards=loss_forwards,
            endpoint_replays=replays,generated_tokens=generated,seconds=time.monotonic()-start,model_loads=1,updates=0,
            base_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),standalone_cost=False))
    except Exception as exc:
        save(a.out/'failed.json',dict(error=repr(exc),answers=answers,visual_calls=vision,loss_forwards=loss_forwards,replays=replays));raise
if __name__=='__main__':main()
