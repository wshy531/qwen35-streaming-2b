"""Source-clustered, paired endpoint analysis; never select checkpoints here."""
import argparse,collections,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def metrics(rows):
    qs=collections.defaultdict(list)
    for r in rows:qs[r['id']].append(r)
    assert all(len(v)==2 and {r['shift'] for r in v}=={0,2} for v in qs.values())
    return dict(questions=len(qs),sources=len({r['video_id'] for r in rows}),answers=len(rows),correct=sum(r['correct'] for r in rows),
        accuracy=sum(r['correct'] for r in rows)/len(rows),both_correct=sum(all(r['correct'] for r in v) for v in qs.values()),
        content_consistent=sum(len({r['choice_original'] for r in v})==1 for v in qs.values()),mean_ce=float(np.mean([r['ce'] for r in rows])),
        by_type={t:dict(correct=sum(r['correct'] for r in rows if r['type'][0]==t),answers=sum(r['type'][0]==t for r in rows)) for t in 'CTD'})
def paired(a,b):
    aa={(r['id'],r['shift']):r for r in a};bb={(r['id'],r['shift']):r for r in b};assert aa.keys()==bb.keys()
    per=collections.defaultdict(list)
    for k,r in aa.items():per[r['video_id']].append(int(bb[k]['correct'])-int(r['correct']))
    sources=sorted(per);num=np.array([sum(per[s]) for s in sources]);den=np.array([len(per[s]) for s in sources])
    rng=np.random.default_rng(20261104);idx=rng.integers(0,len(sources),size=(10000,len(sources)))
    boots=num[idx].sum(1)/den[idx].sum(1)
    return dict(gained=sum(not aa[k]['correct'] and bb[k]['correct'] for k in aa),
        lost=sum(aa[k]['correct'] and not bb[k]['correct'] for k in aa),
        difference_pp=float(num.sum()/den.sum()*100),source_bootstrap95_pp=[float(x*100) for x in np.quantile(boots,[.025,.975])],
        note='conditional on this one training seed; option rotations are clustered within sources')
def main():
    p=argparse.ArgumentParser();p.add_argument('--sequential',type=Path,required=True);p.add_argument('--interleaved',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data={};reports={};train={}
    for arm,path in [('sequential',a.sequential),('interleaved',a.interleaved)]:
        reports[arm]=read(path/'completed.json');assert reports[arm]['status']=='completed'
        assert reports[arm]['questions_trained']==2351 and reports[arm]['updates']==270
        data[arm]=lines(path/'answers.jsonl');updates=lines(path/'updates.jsonl')
        seen=[q['id'] for u in updates for q in u['questions']];assert len(seen)==len(set(seen))==2351
        train[arm]={role:dict(sources=sum(u['role']==role for u in updates),questions=sum(len(u['questions']) for u in updates if u['role']==role),
            mean_online_ce=float(np.mean([q['ce'] for u in updates if u['role']==role for q in u['questions']]))) for role in ('short_train','long_train')}
    initial=lambda rows:{(r['id'],r['shift'],r['view']):r for r in rows if r['stage']=='initial'}
    x,y=initial(data['sequential']),initial(data['interleaved']);assert x.keys()==y.keys()
    baseline_disagreements=[list(k) for k in x if x[k]['choice_original']!=y[k]['choice_original']]
    reference=lines(a.reference/'answers.jsonl') if a.reference else []
    if a.reference:assert read(a.reference/'completed.json')['answers']==1774
    groups={};comparisons={}
    for role,view in [('old_development','full_short'),('long_development','LS'),('long_development','recent8')]:
        key=role+'/'+view;groups[key]={};selected={}
        for arm in data:
            for stage in ('initial','final'):
                rows=[r for r in data[arm] if r['role']==role and r['view']==view and r['stage']==stage]
                name=arm+'/'+stage;groups[key][name]=metrics(rows);selected[name]=rows
        text=[r for r in data['sequential'] if r['role']==role and r['view']=='text' and r['stage']=='initial']
        groups[key]['text']=metrics(text)
        comparisons[key]={arm:paired(selected[arm+'/initial'],selected[arm+'/final']) for arm in data}
        comparisons[key]['interleaved_minus_sequential']=paired(selected['sequential/final'],selected['interleaved/final'])
        if reference:
            rr=[r for r in reference if r['role']==role and r['view']==view]
            groups[key]['old_CE']=metrics(rr)
            comparisons[key]['versus_old_CE']={arm:paired(rr,selected[arm+'/final']) for arm in data}
    history={}
    for arm in data:
        for stage in ('initial','final'):
            rs=[r for r in data[arm] if r['role']=='long_development' and r['stage']==stage]
            history[arm+'/'+stage]=paired([r for r in rs if r['view']=='recent8'],[r for r in rs if r['view']=='LS'])
    if reference:
        history['old_CE']=paired([r for r in reference if r['view']=='recent8'],[r for r in reference if r['view']=='LS'])
    out=dict(reports=reports,baseline_content_disagreements=baseline_disagreements,
        baseline_max_ce_difference=max(abs(x[k]['ce']-y[k]['ce']) for k in x),metrics=groups,paired=comparisons,history_vs_recent=history,
        training=train,limitations=['single seed and one epoch','development not untouched final test','long EOF questions not proven historical dependence',
            'source-average batches; online train CE from different timepoints is not fixed-probe convergence','choice logits classification, no explanation judge or generation'])
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
