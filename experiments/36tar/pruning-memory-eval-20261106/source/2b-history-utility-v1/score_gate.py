"""Measured subset utilities on training sources; no automatic selector result."""
import argparse,gc,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-training-order-main-v2'))
import run_main as m
from resolution_banks_base import Bank
from resolution_nodes import pair_node,digest
from utility_head import subsets
import torch
PREP=ROOT/'runs/history_utility_prepare_20261104_01'

def selected_nodes(recent,history,choice):
    return [history[i] for i in choice]+recent

@torch.no_grad()
def project(model,pair):
    begin,end,pixels,grid=pair;device=model.get_input_embeddings().weight.device
    saved=model.model.rope_deltas
    try:
        output=model.model.get_video_features(pixels.to(device),grid.to(device),return_dict=True)
        features=torch.cat(output.pooler_output,0).detach().float().cpu()
        assert features.shape==(108,model.get_input_embeddings().embedding_dim)
        assert torch.isfinite(features).all()
        return features
    finally:model.model.rope_deltas=saved

@torch.no_grad()
def forward(model,builder,nodes,text,representation=False):
    emb,pos,ids,meta=m.context(model,builder,nodes,text,visual_limit=864)
    hidden=model.model.language_model(inputs_embeds=emb,position_ids=pos.clone(),
        attention_mask=torch.ones(emb.shape[:2],device=emb.device,dtype=torch.long),use_cache=False).last_hidden_state[:,-1]
    if representation:return hidden[0].float().cpu()
    logits=model.lm_head(hidden).float()[0];assert torch.isfinite(logits).all()
    return logits,ids.numel()

@torch.no_grad()
def answer(model,builder,nodes,q,shift):
    rq=m.rotate(q,shift);logits,tokens=forward(model,builder,nodes,m.question_text(rq))
    ids=m.option_ids(builder,logits.device);values=logits[ids];p=values.softmax(-1);gold=rq['answer_index']
    choice=int(values.argmax())
    return dict(id=q['id'],video_id=q['video_id'],type=q['type'],shift=shift,gold_index=gold,
        option_logits=values.cpu().tolist(),gold_probability=float(p[gold]),
        full_vocab_gold_probability=float(logits.softmax(-1)[ids[gold]]),
        ce=float(-logits.log_softmax(-1)[ids[gold]]),correct=choice==gold,
        choice_original=(choice+shift)%5,input_tokens=tokens,visual_tokens=sum(len(n.features) for n in nodes))

def descriptor(nodes):
    x=torch.cat([n.features for n in nodes]);return torch.cat((x.mean(0),x.amax(0)))

def features(model,builder,recent,history,q):
    # Deliberately question-only: identical query features for all option orders.
    query=forward(model,builder,recent,q['question'],representation=True)
    width=model.get_input_embeddings().embedding_dim
    hist=torch.zeros(4,2*width);meta=torch.zeros(4,8);mask=torch.zeros(4,dtype=torch.bool)
    now=max(recent[-1].end,1e-6);pairs=recent[-1].last+1
    for i,n in enumerate(history):
        hist[i]=descriptor([n]);mask[i]=True
        meta[i]=torch.tensor([n.begin/now,n.end/now,n.center/now,(n.end-n.begin)/now,
            (n.last-n.first+1)/pairs,len(n.features)/108,n.depth/max(pairs,1),n.groups/max(pairs,1)])
    return dict(id=q['id'],video_id=q['video_id'],query=query,recent=descriptor(recent),
        history=hist,metadata=meta,node_mask=mask)

def synthetic_bank(group,width):
    bank=Bank('LS',108,budget=864,recent=432,closed=324)
    for i in range(group['audit']['encoding_pairs']):
        bank.append(pair_node(torch.zeros(108,width),i,2.*i,2.*i+1,(9,12)),starts_event=i%3==0)
    return bank

def layouts(model,builder,groups):
    total=maximum=0
    for g in groups:
        bank=synthetic_bank(g,model.get_input_embeddings().embedding_dim)
        history=[e.node for e in bank.history];recent=[e.node for e in bank.recent]
        assert len(history)==4 and len(recent)==4
        choices=subsets(len(history))
        for q in g['questions']:
            for choice in choices:
                nodes=selected_nodes(recent,history,choice)
                for shift in (0,2):
                    _,_,ids,_=m.context(model,builder,nodes,m.question_text(m.rotate(q,shift)),visual_limit=864)
                    total+=1;maximum=max(maximum,ids.numel())
            # All possible selected choices for held rotations are checked,
            # rather than pretending the not-yet-known winner is available.
            for choice in choices:
                for shift in (1,3):
                    _,_,ids,_=m.context(model,builder,selected_nodes(recent,history,choice),m.question_text(m.rotate(q,shift)),visual_limit=864)
                    total+=1;maximum=max(maximum,ids.numel())
            m.context(model,builder,recent,q['question'],visual_limit=864)
    return dict(layouts=total,query_layouts=sum(len(g['questions']) for g in groups),max_tokens=maximum)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cpu-check',action='store_true');a=ap.parse_args()
    assert a.out.resolve().is_relative_to((ROOT/'runs').resolve()) and not a.out.exists()
    lock=m.read(PREP/'gate_lock.json')
    for p,h in lock.items():assert m.sha(ROOT/p)==h,p
    groups=[g for g in m.read(PREP/'train_manifest_LOCAL_ONLY.json') if g['gate_batch']]
    assert len(groups)==32 and sum(len(g['questions']) for g in groups)==314
    assert sum(g['audit']['encoding_pairs'] for g in groups)==1729
    torch.set_num_threads(1);torch.manual_seed(20261104)
    if a.cpu_check:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
        assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>20*2**30
    a.out.mkdir();start=time.monotonic();env=m.runtime()
    b=m.make_builder(env,env['AutoProcessor'].from_pretrained(m.WEIGHTS,local_files_only=True))
    if a.cpu_check:
        config=env['tiny_config'](b.processor);config['vision_config']['depth']=24
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config,attn_implementation='sdpa')).float()
    else:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(m.WEIGHTS,local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
    counts=dict(sources=0,questions=0,classifications=0,representation_forwards=0,visual_calls=0)
    def guard():
        if time.monotonic()-start>88*60:raise TimeoutError('fixed 88 minute stop; no automatic continuation')
        assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<96*2**20
    if a.cpu_check:
        with env['fp32_reference']():
            layout=layouts(model,b,groups)
            pairs=m.load_pairs(env,b,groups[0])[:10];bank=Bank('LS',108)
            for i,pair in enumerate(pairs):bank.append(pair_node(project(model,pair),i,pair[0],pair[1],(9,12)),starts_event=i%3==0)
            history=[e.node for e in bank.history];recent=[e.node for e in bank.recent];before=digest(bank.nodes())
            choices=subsets(len(history));q=groups[0]['questions'][0]
            rows=[answer(model,b,selected_nodes(recent,history,c),q,s) for c in ((),(0,),tuple(range(4))) for s in (0,2)]
            f=features(model,b,recent,history,q)
            assert digest(bank.nodes())==before and len(choices)==16
            assert torch.isfinite(f['query']).all()
        m.save(a.out/'completed.json',dict(status='cpu_passed',**layout,real_visual_pairs=10,micro_answers=len(rows),
            bank_unchanged=True,base_unchanged=all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters()),seconds=time.monotonic()-start))
        return
    with env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as log,(a.out/'decisions.jsonl').open('x',encoding='utf-8') as decisions:
        for g in groups:
            guard();pairs=m.load_pairs(env,b,g);bank=Bank('LS',108)
            for i,pair in enumerate(pairs):
                guard();bank.append(pair_node(project(model,pair),i,pair[0],pair[1],(9,12)),starts_event=i%3==0);counts['visual_calls']+=1
            history=[e.node for e in bank.history];recent=[e.node for e in bank.recent];before=digest(bank.nodes())
            assert len(history)==4 and len(recent)==4
            choices=subsets(4);source_features=[]
            for q in g['questions']:
                guard();measured=[]
                for index,choice in enumerate(choices):
                    rows=[]
                    for shift in (0,2):
                        row=answer(model,b,selected_nodes(recent,history,choice),q,shift)
                        log.write(json.dumps(dict(row,phase='fit',subset=index,history_indices=choice))+'\n');rows.append(row);counts['classifications']+=1
                    measured.append(sum(r['gold_probability'] for r in rows)/2)
                winner=max(range(16),key=lambda i:(measured[i],-len(choices[i]),-i))
                # Commit the selection BEFORE any held-rotation forward.
                decisions.write(json.dumps(dict(id=q['id'],video_id=g['video_id'],selected=winner,
                    fit_probabilities=measured,fit_gains=[v-measured[0] for v in measured],diagnostic_oracle=True))+'\n');decisions.flush()
                for name,index in (('selected',winner),('recent',0),('all',15)):
                    for shift in (1,3):
                        row=answer(model,b,selected_nodes(recent,history,choices[index]),q,shift)
                        log.write(json.dumps(dict(row,phase='confirmation',view=name,subset=index,history_indices=choices[index]))+'\n');counts['classifications']+=1
                source_features.append(features(model,b,recent,history,q));counts['representation_forwards']+=1;counts['questions']+=1
                log.flush()
            assert digest(bank.nodes())==before
            torch.save(dict(rows=source_features,history=[n.metadata() for n in history],recent=[n.metadata() for n in recent],
                bank_sha256=before,source_sha256=g['sha256']),a.out/f"features-{g['video_id']}.pt")
            counts['sources']+=1
            print(json.dumps(dict(**counts,seconds=time.monotonic()-start)),flush=True)
            del source_features,pairs,bank,history,recent;gc.collect()
    assert counts==dict(sources=32,questions=314,classifications=11932,representation_forwards=314,visual_calls=1729),counts
    assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
    for p,h in lock.items():assert m.sha(ROOT/p)==h,p
    m.save(a.out/'completed.json',dict(status='completed',**counts,seconds=time.monotonic()-start,generated_tokens=0,updates=0,
        base_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
