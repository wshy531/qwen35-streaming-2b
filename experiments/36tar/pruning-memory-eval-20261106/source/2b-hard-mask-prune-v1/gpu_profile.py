"""Three train clips, two objective arms: six updates, no generation or capability claim."""
import argparse, gc, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/2b-learned-prune-main-v1'))
from protocol import runtime, make_builder, decode, read, sha, DATA, rotate, train_shift, encode_pair, CoverageScorer
from hard_prune import torch, GroupScorer, encode
from objective import last_logits, option_ids, loss_parts
from qwen_mask_check import check_visual
PREP = ROOT/'runs/hard_mask_prepare_20261102_01'


def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out', type=Path, required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
    for name, digest in read(PREP/'execution_lock.json').items():
        assert sha(ROOT/name) == digest, name
    assert not a.out.exists() and a.out.resolve().is_relative_to((ROOT/'runs').resolve())
    a.out.mkdir();started=time.monotonic();updates=visual_calls=0
    def guard():
        if time.monotonic()-started > 12*60:raise TimeoutError('fixed profile deadline')
        assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file()) < 32*2**20
    try:
        env=runtime();torch.set_num_threads(1)
        assert torch.cuda.device_count()==1 and env['psutil'].virtual_memory().available >= 12*2**30
        model, info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
            local_files_only=True, dtype=torch.float32, device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
        builder=make_builder(env, env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        rows=read(ROOT/'runs/learned_prune_profile_prepare_20261101_01/selection.json')
        audit={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
        cases=[]
        for row in rows:
            assert row['source']['role']==row['question']['role']=='train'
            path=ROOT/'datasets/learned-prune-nextqa-v1'/row['source']['mirror_path']
            assert sha(path)==row['source']['mirror_lfs_sha256']
            cases.append((rotate(row['question'],train_shift(row['question'])),
                          decode(env,path,audit[row['source']['video_id']],builder)))
        assert len(cases)==3 and sum(len(pairs) for _,pairs in cases)==44
        scorer=GroupScorer(model.config.vision_config.hidden_size).to('cuda')
        initial=torch.load(ROOT/'runs/learned-prune-main-220024/scorer_untrained.pt',map_location='cpu',weights_only=True)
        scorer.load_state_dict(initial)
        first=cases[0][1][0][2].to('cuda')
        with env['fp32_reference']():
            alignment,_=check_visual(model.model.visual,first.pixel_values_videos,first.video_grid_thw,
                                      scorer,atol=2e-4,rtol=5e-5)
        visual_calls+=7 # conservative: one prefix plus six tails counted as full calls
        save(a.out/'visual_alignment.json',alignment)
        del first;gc.collect();torch.cuda.empty_cache()
        options=option_ids(builder,torch.device('cuda'))
        teachers=[]
        with torch.no_grad(),env['fp32_reference']():
            for q,pairs in cases:
                guard();records=[]
                for i,(begin,end,batch) in enumerate(pairs):
                    batch=batch.to('cuda')
                    feat,ids,_=encode_pair(model,batch,'full',None,None,CoverageScorer())
                    records.append((i,begin,end,feat,ids));visual_calls+=1
                logits,tokens=last_logits(model,builder,records,q)
                teachers.append(logits[:,options].detach().cpu())
                del records,logits;gc.collect()
        steps=[];logit_checks=[]
        with (a.out/'steps.jsonl').open('x',encoding='utf-8') as stream:
            for arm in ('ce','ce_kl'):
                scorer.load_state_dict(initial)
                opt=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
                for ci,(q,pairs) in enumerate(cases):
                    guard();opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
                    records=[];state=None;before={n:p.detach().clone() for n,p in scorer.named_parameters()}
                    with env['fp32_reference']():
                        for i,(begin,end,batch) in enumerate(pairs):
                            batch=batch.to('cuda')
                            feat,ids,_,state=encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,
                                                  scorer,previous=state,mode='mask')
                            records.append((i,begin,end,feat,ids));visual_calls+=1
                        logits,tokens=last_logits(model,builder,records,q)
                        loss,ce,kl,active=loss_parts(logits,options,q['answer_index'],
                                                    teachers[ci].to('cuda') if arm=='ce_kl' else None)
                        assert torch.isfinite(loss);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.)
                    assert torch.isfinite(grad) and grad>0
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scorer.parameters())
                    assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
                    torch.cuda.synchronize();seconds=time.monotonic()-tick
                    peak=torch.cuda.max_memory_allocated()
                    # Compare the same PRE-update checkpoint to physical gather after freeing graph.
                    expected=logits.detach().clone();ce_value=float(ce.detach());kl_value=float(kl.detach())
                    loss_value=float(loss.detach())
                    del records,logits,loss,ce,kl,feat;gc.collect()
                    with torch.no_grad(),env['fp32_reference']():
                        records=[];state=None
                        for i,(begin,end,batch) in enumerate(pairs):
                            batch=batch.to('cuda')
                            feat,ids,_,state=encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,
                                                  scorer,previous=state,mode='gather')
                            records.append((i,begin,end,feat,ids));visual_calls+=1
                        actual,_=last_logits(model,builder,records,q)
                        delta=actual-expected
                        check=dict(arm=arm,id=q['id'],max_abs=float(delta.abs().max()),
                            relative_l2=float(delta.norm()/expected.norm().clamp_min(1e-12)),
                            argmax_equal=bool(actual.argmax(-1)==expected.argmax(-1)),
                            option_rank_equal=torch.equal(actual[:,options].argsort(),expected[:,options].argsort()))
                        logit_checks.append(check);save(a.out/'language_alignment.json',logit_checks)
                        assert check['max_abs']<1e-3 and check['relative_l2']<5e-5 and check['argmax_equal'] and check['option_rank_equal'],check
                        del records,actual,expected,feat
                    opt.step();updates+=1
                    assert any(not torch.equal(p,before[n]) for n,p in scorer.named_parameters())
                    entry=dict(arm=arm,id=q['id'],pairs=len(pairs),input_tokens=tokens,loss=loss_value,ce=ce_value,
                               kl=kl_value,teacher_correct=bool(int(teachers[ci].argmax(-1))==q['answer_index']),
                               kl_active=active,grad_norm=float(grad),seconds=seconds,peak_allocated_bytes=peak)
                    steps.append(entry);stream.write(json.dumps(entry)+'\n');stream.flush();print(json.dumps(entry),flush=True)
                    del before;gc.collect();torch.cuda.empty_cache()
                torch.save(scorer.state_dict(),a.out/f'scorer_{arm}_profile_only.pt')
                del opt
        assert updates==6 and visual_calls==227
        save(a.out/'completed.json',dict(status='passed_interface_not_capability',updates=updates,steps=steps,
            vision_call_upper_count=visual_calls,generated_tokens=0,base_unchanged=True,model_loads=1,
            trainable_parameters=sum(p.numel() for p in scorer.parameters()),seconds=time.monotonic()-started))
    except Exception as e:
        save(a.out/'failed.json',dict(error=repr(e),updates=updates,vision_call_upper_count=visual_calls));raise


if __name__=='__main__':main()
