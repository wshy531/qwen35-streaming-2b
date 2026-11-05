"""Fixed CPU training on exported descriptors, no base model or hyperparameter search."""
import argparse,json,copy,time,os
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from prepare_data import ROOT,PREP,read,save
from selector_model import EvidenceSelector,objective,select

def training_targets(ids,traces):
    short=read(PREP/'supervision_LOCAL_ONLY.json');long=read(PREP/'long_supervision_LOCAL_ONLY.json')
    bytrace={r['id']:r for r in traces};meta=[]
    for tid in ids:
        s=short.get(tid,long.get(tid));h=bytrace[tid]['history']
        if tid in short:
            positive=s['positive_history_indices'];group=s['source_group']
        else:
            positive=[i for i,n in enumerate(h) if any(max(n['begin'],a)<=min(n['end'],b) for a,b in s['intervals'])]
            # Broad intervals crossing most nodes do not teach useful selectivity.
            if len(positive)>2:positive=[]
            group='sember:'+next(r['parent_source'] for r in read(PREP/'long_requests.json') if r['id']==tid)
        if any(i>=len(h) for i in positive):raise ValueError('citation mapping beyond actual bank')
        meta.append(dict(id=tid,role=s['role'],group=group,route_known=s['route_known'],
            needs_history=s.get('needs_history',0),positive=positive,long_proxy=tid in long))
    return meta

def main(features,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' or out.exists():raise RuntimeError('CPU-only/new output required')
    torch.set_num_threads(1);started=time.monotonic();out.mkdir()
    with np.load(features/'features.npz',allow_pickle=False) as f:
        ids=f['ids'].tolist();arrays={k:torch.from_numpy(f[k].copy()) for k in ('query','recent','history','metadata','mask')}
    assert len(ids)==455 and arrays['query'].shape==(455,2048)
    rows=training_targets(ids,read(features/'nodes.json'));train=[i for i,r in enumerate(rows) if r['role']=='train']
    counts=Counter(rows[i]['group'] for i in train);raw={i:1/counts[rows[i]['group']] for i in train}
    masses={v:sum(raw[i] for i in train if rows[i]['route_known'] and rows[i]['needs_history']==v) for v in (0,1)}
    assert all(v>0 for v in masses.values())
    route_weights=[(.5*raw[i]/masses[rows[i]['needs_history']] if i in raw and rows[i]['route_known'] else 1.) for i in range(len(ids))]
    rank_weights=[((.25 if rows[i]['long_proxy'] else 1.)*raw.get(i,1.)) for i in range(len(ids))]
    positive=torch.zeros((len(ids),4),dtype=torch.bool)
    for i,r in enumerate(rows):positive[i,r['positive']]=True
    arrays.update(route_known=torch.tensor([r['route_known'] for r in rows]),needs_history=torch.tensor([r['needs_history'] for r in rows],dtype=torch.float32),
        route_weight=torch.tensor(route_weights),rank_weight=torch.tensor(rank_weights),positive_mask=positive)
    results={};models=[]
    for blind in (False,True):
        torch.manual_seed(20261029);model=EvidenceSelector();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        rng=torch.Generator().manual_seed(20261029);updates=0;loss_trace=[]
        for epoch in range(20):
            model.train();order=torch.tensor(train)[torch.randperm(len(train),generator=rng)];epoch_losses=[]
            for ix in order.split(16):
                optimizer.zero_grad();loss,_=objective(model,{k:v[ix] for k,v in arrays.items()},blind_history=blind)
                if not loss.requires_grad:continue
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite selector loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1;epoch_losses.append(float(loss.detach()))
            loss_trace.append(float(np.mean(epoch_losses)))
            if time.monotonic()-started>240:raise TimeoutError('bounded CPU training exceeded 4 minutes')
        model.eval();predictions=[]
        with torch.no_grad():
            for i,r in enumerate(rows):
                logits,route=model(*(arrays[k][i:i+1] for k in ('query','recent','history','metadata','mask')),blind_history=blind)
                n=int(arrays['mask'][i].sum());decision=select(logits[0,:n+1],route[0],[108]*n)
                predictions.append(dict(id=ids[i],role=r['role'],reading=decision['reading'],history_indices=decision['history_indices'],
                    route_logit=float(route[0]),weak_citation_hit=bool(set(decision['history_indices'])&set(r['positive'])),
                    has_citation_supervision=bool(r['positive']),route_target=r['needs_history'] if r['route_known'] else None))
        name='history_blind' if blind else 'content_visible'
        torch.save(dict(state_dict=model.state_dict(),architecture='EvidenceSelector64_v1',history_blind=blind,
            train_ids=[ids[i] for i in train],parameters=sum(p.numel() for p in model.parameters()),epochs=20,seed=20261029),out/(name+'.pt'))
        save(out/(name+'_predictions.json'),predictions)
        summary={}
        for role in sorted({r['role'] for r in rows}):
            part=[p for p in predictions if p['role']==role];known=[p for p in part if p['route_target'] is not None]
            summary[role]=dict(tasks=len(part),route_labelled=len(known),route_agreement=sum((p['reading']=='history')==bool(p['route_target']) for p in known),
                chose_history=sum(p['reading']=='history' for p in part),
                history_needed_recalled=sum(p['route_target']==1 and p['reading']=='history' for p in known),
                history_needed_denominator=sum(p['route_target']==1 for p in known),
                weak_citation_hits=sum(p['weak_citation_hit'] for p in part if p['has_citation_supervision']))
        results[name]=dict(updates=updates,epoch_losses=loss_trace,summary=summary)
    save(out/'fit_analysis.json',dict(results=results,training_examples=len(train),
        short_route_supervised_train=sum(rows[i]['route_known'] for i in train),long_rank_only_train=sum(rows[i]['long_proxy'] for i in train),
        long_train_with_usable_proxy=sum(rows[i]['long_proxy'] and bool(rows[i]['positive']) for i in train),
        no_new_QA_answers=True,no_base_updates=True,opened_development_material=True,formal_generalization_claim=False,
        inference_extra_question_prefill_cost_not_measured=True,CPU_seconds=time.monotonic()-started))
    print('Two fixed selector heads fitted. These are teacher/proxy agreement results, not QA accuracy.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.features,a.out)
