"""Fixed CPU training; evaluation labels are accessed only after weights freeze."""
from collections import Counter,defaultdict
from common import *
from selector import Head,choices_mask,choose,objective
from temporal_policy import lexical_route

FIELDS=('query','recent','history','metadata','node_mask')
def fit_and_evaluate(out):
    torch.set_num_threads(1)
    rows=torch.load(out/'features.pt',map_location='cpu',weights_only=True)
    scores=[json.loads(s) for s in (out/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
    bykey={(r['id'],r['subset'],r['shift']):r for r in scores}
    assert len(bykey)==len(scores)
    gold={q['id']:q['answer_index'] for g in read(PREP/'fit_sources.json') for q in g['questions']}
    train=[f for f in rows if f['role']=='train'];dev=[f for f in rows if f['role']=='development'];evaluation=[f for f in rows if f['role']=='evaluation']
    assert (len(train),len(dev),len(evaluation))==(144,32,288)
    assert len({f['source_id'] for f in train})==72
    assert not ({f['source_id'] for f in train}&{f['source_id'] for f in dev+evaluation})
    choices,mask=choices_mask(4)
    def pack(fs):
        batch=[torch.stack([f[k] for f in fs]) for k in FIELDS]
        candidates=torch.zeros(len(fs),16,4,dtype=torch.bool);valid=torch.zeros(len(fs),16,dtype=torch.bool)
        target=torch.zeros(len(fs),16)
        for j,f in enumerate(fs):
            ps=[]
            for i,c in enumerate(f['choices']):
                candidates[j,i,c]=True;valid[j,i]=True
                vals=[]
                for shift in (0,2):
                    r=bykey[f['id'],i,shift];logit=torch.tensor(r['option_logits'])
                    vals.append(logit.softmax(-1)[(gold[f['id']]-shift)%len(logit)])
                ps.append(float(sum(vals)/2))
            target[j,:len(ps)]=torch.tensor(ps)-ps[0]
        return batch+[candidates],valid,target
    batch,valid,target=pack(train);weights=torch.ones(len(train))
    heads={};training={}
    for name,blind in [('visible',False),('blind',True)]:
        torch.manual_seed(20261104);model=Head();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
        gen=torch.Generator().manual_seed(20261104);history=[]
        for epoch in range(1,21):
            order=torch.randperm(len(train),generator=gen);losses=[]
            for start in range(0,len(order),16):
                idx=order[start:start+16];opt.zero_grad(set_to_none=True)
                pred=model(*[x[idx] for x in batch],blind_history=blind)
                loss=objective(pred,target[idx],valid[idx],weights[idx]);assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,mean_huber=sum(losses)/len(losses),last_grad_norm=float(norm)))
        model.eval().requires_grad_(False);heads[name]=model;training[name]=history
        torch.save(dict(state_dict=model.state_dict(),width=3584,epoch=20,train_questions=144,train_sources=72,blind_history=blind,seed=20261104),out/f'head-{name}.pt')
    save(out/'training.json',training)
    # Only now open evaluation answers; they cannot affect parameters or routing.
    gold.update(read(PREP/'evaluation_keys_LOCAL_ONLY.json'))
    allrows=[];selections=[]
    with torch.no_grad():
        for f in dev+evaluation:
            choices,mask=choices_mask(int(f['node_mask'].sum()));n=len(choices[-1]);routes={'recent':0,'LS':len(choices)-1,
                'lexical':len(choices)-1 if lexical_route(f['question'])['route']=='full' else 0}
            for name,h in heads.items():
                values=h(*(f[k][None] for k in FIELDS),mask,blind_history=name=='blind')[0]
                chosen=choose(values,choices);routes[name]=chosen;k=len(choices[chosen])
                routes[name+'_same_count_earliest']=choices.index(list(range(k)))
                routes[name+'_same_count_latest']=choices.index(list(range(n-k,n)))
                selections.append(dict(id=f['id'],role=f['role'],head=name,choices=choices,predicted_gains=values.tolist(),selected=choices[chosen]))
            for name,i in routes.items():
                for shift in (0,2):
                    r=bykey[f['id'],i,shift]
                    allrows.append(dict(id=f['id'],source_id=f['source_id'],suite=f['suite'],role=f['role'],method=name,
                        shift=shift,selected=r['selected_original'],gold=gold[f['id']],correct=r['selected_original']==gold[f['id']],
                        history_nodes=choices[i],visual_tokens=r['visual_tokens']))
    save(out/'selected_routes.json',selections);save(out/'scored.json',allrows)
    tables=[]
    for suite in sorted({r['suite'] for r in allrows}):
        for method in sorted({r['method'] for r in allrows}):
            rs=[r for r in allrows if r['suite']==suite and r['method']==method]
            tables.append(dict(suite=suite,method=method,correct=sum(r['correct'] for r in rs),n=len(rs),
                questions=len({r['id'] for r in rs}),mean_visual_tokens=sum(r['visual_tokens'] for r in rs)/len(rs)))
    save(out/'analysis.json',dict(tables=tables,protocol='normalized option-score selection, two permutations; not generation accuracy',
        migration='recipe refitted at new width; not zero-shot old-checkpoint transfer',train_questions=144,development_questions=32,evaluation_questions=288))
    return tables
