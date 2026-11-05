"""Full official-supervision curriculum comparison; source-batched gradients."""
import argparse, gc, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-training-order-v1'))
from run_order import (WEIGHTS, GroupScorer, capture_pairs, context, decode, last_logits,
    loss_parts, make_builder, option_ids, qa_leaf_gradients, question_text, read,
    rotate, runtime, sha, snapshot, train_shift, visual_backward)
from adapter import sparse_pair
import torch
PREP=ROOT/'runs/training_order_main_prepare_20261104_02'

def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def load_pairs(env,builder,group):
    path=ROOT/group['path']
    assert sha(path)==group['sha256'],group['video_id']
    batches=decode(env,path,group['audit'],builder)
    assert len(batches)==group['audit']['encoding_pairs']
    return [(begin,end,batch.pixel_values_videos,batch.video_grid_thw) for begin,end,batch in batches]

def all_nodes(captured,pairs):
    return [sparse_pair(p.features,p.ids,i,pairs[i][0],pairs[i][1]) for i,p in captured.pairs.items()]

def capture(model,scorer,pairs,short):
    c=capture_pairs(model.model.visual,scorer,pairs)
    if short: c.nodes=all_nodes(c,pairs)
    return c

def qa_leaf_gradients(model,builder,captured,question,answer_index):
    # Full short clips have at most 20 pairs (1080 tokens). The long bank
    # remains LS864. Set the readout allowance explicitly without truncation.
    device=model.get_input_embeddings().weight.device
    leaves={i:p.features.to(device).detach().clone().requires_grad_(True) for i,p in captured.pairs.items()}
    emb,pos,ids,meta=context(model,builder,captured.nodes,question,visual_limit=1080)
    visual=torch.cat(captured.replay.values(captured.nodes,leaves)).to(emb)
    mask=(ids.to(device)==builder.processor.video_token_id).unsqueeze(-1).expand_as(emb)
    assert int(mask.sum())==visual.numel()
    emb=emb.clone().masked_scatter(mask,visual)
    logits=last_logits(model,emb,pos.clone())
    loss=loss_parts(logits,option_ids(builder,device),answer_index)[0];loss.backward()
    grads={i:v.grad.detach().cpu().clone() for i,v in leaves.items() if v.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    return dict(loss=float(loss.detach()),gradients=grads)

def batch_gradient(model,builder,scorer,captured,pairs,questions):
    """Average QA gradients at fixed scorer weights, then one visual replay.

    Equivalent to accumulating per-question scorer gradients at these weights
    up to floating point order; optimizer never steps inside this function.
    """
    aggregate={}; rows=[]
    for q in questions:
        rq=rotate(q,train_shift(q))
        qa=qa_leaf_gradients(model,builder,captured,question_text(rq),rq['answer_index'])
        for index,grad in qa['gradients'].items():
            if index not in aggregate: aggregate[index]=grad/len(questions)
            else: aggregate[index].add_(grad,alpha=1/len(questions))
        rows.append(dict(id=q['id'],ce=qa['loss'],shift=train_shift(q)))
    gradients=visual_backward(model.model.visual,scorer,captured,aggregate,pairs.__getitem__)
    return rows,len(aggregate),sum(float(g.abs().sum()) for g in gradients.values())

@torch.no_grad()
def answer(model,builder,q,nodes,shift):
    rq=rotate(q,shift);prompt=question_text(rq);device=model.get_input_embeddings().weight.device
    if nodes is None:
        rendered=builder.template(prompt);assert rendered.count(builder.marker)==1
        ids=builder.tokenize(rendered.replace(builder.marker,'',1)).to(device)
        assert not (ids==builder.processor.video_token_id).any()
        emb=model.get_input_embeddings()(ids);pos=torch.arange(ids.shape[1],device=device).expand(4,1,-1)
    else: emb,pos,ids,_=context(model,builder,nodes,prompt,visual_limit=1080)
    logits=last_logits(model,emb,pos);options=option_ids(builder,device)
    choice=int(logits[0,options].argmax())
    return dict(id=q['id'],video_id=q['video_id'],type=q['type'],shift=shift,
        correct=choice==rq['answer_index'],choice_original=(choice+shift)%5,
        ce=float(loss_parts(logits,options,rq['answer_index'])[1]),tokens=ids.numel())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['sequential','interleaved'],required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cpu-check',action='store_true');args=ap.parse_args()
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve()) and not args.out.exists()
    for path,digest in read(PREP/'execution_lock.json').items():assert sha(ROOT/path)==digest,path
    torch.set_num_threads(1);torch.manual_seed(20261104)
    if args.cpu_check: assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
        assert torch.cuda.device_count()==1
    args.out.mkdir();start=time.monotonic();panel=read(PREP/'panel_LOCAL_ONLY.json');groups=panel['groups']
    env=runtime();builder=make_builder(env,env['AutoProcessor'].from_pretrained(WEIGHTS,local_files_only=True))
    if args.cpu_check:
        config=env['tiny_config'](builder.processor);config['vision_config']['depth']=24
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config,attn_implementation='sdpa')).float()
    else:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(WEIGHTS,local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
    device=model.get_input_embeddings().weight.device
    torch.manual_seed(20261104);scorer=GroupScorer(model.config.vision_config.hidden_size).to(device)
    initial=snapshot(scorer);torch.save(initial,args.out/'initial.pt')
    if not args.cpu_check: assert sum(p.numel() for p in scorer.parameters())==69697
    counts=dict(questions_trained=0,updates=0,visual_calls=0,answers=0)
    def guard():
        if time.monotonic()-start>88*60: raise TimeoutError('fixed 88 minute stop')
        assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())<128*2**20
    if args.cpu_check:
        # One real long-source batch checks the only new autograd operation:
        # sum at projected leaves versus sum at scorer parameters.
        group=next(g for g in groups.values() if g['role']=='long_train' and g['audit']['encoding_pairs']>=18)
        pairs=load_pairs(env,builder,group)[:18];qs=group['questions'][:2]
        with env['fp32_reference']():
            c=capture(model,scorer,pairs,False);scorer.zero_grad(set_to_none=True)
            batch_gradient(model,builder,scorer,c,pairs,qs)
            combined={n:p.grad.clone() for n,p in scorer.named_parameters()}
            scorer.zero_grad(set_to_none=True)
            for q in qs:
                rq=rotate(q,train_shift(q));qa=qa_leaf_gradients(model,builder,c,question_text(rq),rq['answer_index'])
                visual_backward(model.model.visual,scorer,c,{i:g/len(qs) for i,g in qa['gradients'].items()},pairs.__getitem__)
            for n,p in scorer.named_parameters():torch.testing.assert_close(combined[n],p.grad,atol=1e-7,rtol=1e-4)
            delta=max(float((combined[n]-p.grad).abs().max()) for n,p in scorer.named_parameters())
            before=snapshot(scorer);opt=torch.optim.AdamW(scorer.parameters(),lr=1e-4)
            torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.);opt.step()
            assert any(not torch.equal(before[n],p.cpu()) for n,p in scorer.named_parameters())
            c=capture(model,scorer,pairs,False)
            recent=all_nodes(c,pairs)[-8:]
            check=[answer(model,builder,qs[0],nodes,0) for nodes in (c.nodes,recent,None)]
            # Cover the failure boundary explicitly: 18 complete pairs >864.
            c.nodes=all_nodes(c,pairs)
            check.append(answer(model,builder,qs[0],c.nodes,0))
            scorer.zero_grad(set_to_none=True)
            batch_gradient(model,builder,scorer,c,pairs,qs[:1])
        save(args.out/'completed.json',dict(status='cpu_passed',pairs=18,questions=2,merged_nodes=sum(n.first<n.last for n in c.nodes),
            gradient_max_delta=delta,
            answers=check,seconds=time.monotonic()-start));return
    def evaluate(stage,log):
        for index,vid in enumerate(panel['development']):
            guard();g=groups[vid];pairs=load_pairs(env,builder,g);short=g['role']=='old_development'
            c=capture(model,scorer,pairs,short);counts['visual_calls']+=len(pairs)
            views={'full_short' if short else 'LS':c.nodes}
            if not short:views['recent8']=all_nodes(c,pairs)[-8:]
            if stage=='initial':views['text']=None
            for view,nodes in views.items():
                for q in g['questions']:
                    for shift in (0,2):
                        r=answer(model,builder,q,nodes,shift);counts['answers']+=1
                        log.write(json.dumps(dict(r,stage=stage,view=view,role=g['role'],arm=args.arm))+'\n')
            log.flush();del pairs,c;gc.collect()
            if (index+1)%10==0:print(json.dumps(dict(stage=stage,evaluation_sources=index+1,**counts,seconds=time.monotonic()-start)),flush=True)
    try:
        with env['fp32_reference'](),(args.out/'answers.jsonl').open('x',encoding='utf-8') as answers,(args.out/'updates.jsonl').open('x',encoding='utf-8') as log:
            evaluate('initial',answers)
            optimizer=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
            for step,vid in enumerate(panel['schedules'][args.arm],1):
                guard();tick=time.monotonic();g=groups[vid];pairs=load_pairs(env,builder,g)
                optimizer.zero_grad(set_to_none=True);c=capture(model,scorer,pairs,g['role']=='short_train')
                rows,recomputed,grad_l1=batch_gradient(model,builder,scorer,c,pairs,g['questions'])
                norm=torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.,error_if_nonfinite=True)
                assert float(norm)>0 and all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
                optimizer.step();counts['updates']+=1;counts['questions_trained']+=len(rows);counts['visual_calls']+=len(pairs)+recomputed
                log.write(json.dumps(dict(step=step,video_id=vid,role=g['role'],questions=rows,pairs=len(pairs),
                    mean_ce=sum(r['ce'] for r in rows)/len(rows),grad_norm=float(norm),score_grad_l1=grad_l1,
                    merged_nodes=sum(n.first<n.last for n in c.nodes),seconds=time.monotonic()-tick))+'\n');log.flush()
                if step in (68,135,150,203,270):
                    torch.save(snapshot(scorer),args.out/f'scorer_step{step}.pt')
                    torch.save(dict(optimizer=optimizer.state_dict(),step=step),args.out/'optimizer_latest.pt')
                del pairs,c;gc.collect()
                if step%10==0: print(json.dumps(dict(stage='train',**counts,seconds=time.monotonic()-start)),flush=True)
            assert counts['updates']==270 and counts['questions_trained']==2351
            torch.save(snapshot(scorer),args.out/'final.pt')
            evaluate('final',answers)
        assert counts['answers']==4780
        assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
        save(args.out/'completed.json',dict(status='completed',arm=args.arm,**counts,seconds=time.monotonic()-start,
            generated_tokens=0,base_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            protocol_sha256=sha(PREP/'protocol.json'),panel_sha256=sha(PREP/'panel_LOCAL_ONLY.json')))
    except Exception as e:
        torch.save(snapshot(scorer),args.out/'interrupted.pt')
        save(args.out/'failed.json',dict(error=repr(e),**counts,seconds=time.monotonic()-start));raise
if __name__=='__main__':main()
