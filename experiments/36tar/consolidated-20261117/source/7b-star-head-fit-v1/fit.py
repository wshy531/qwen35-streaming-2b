import sys,json,hashlib,re,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/7b-pruned-head-adapt-v1'))
from adapt import Head,FIELDS,objective,torch,choose
OUT=ROOT/'runs/7b_star_head_fit_20261113_01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load():
    old=ROOT/'runs/7b-pruned-head-adapt-223422/snapshot-01';star=ROOT/'runs/7b-star-utility-223721';data=ROOT/'runs/head_data_coverage_20261112_01'
    assert sha(star/'features.pt')=='b70971e40930e148a780b9a2020e507131c2f44d357395c952ae677abf05cc61'
    oq={q['id']:q for g in read(ROOT/'runs/7b_pruned_head_adapt_prepare_20261111_01/fit_sources.json') for q in g['questions']}
    sq={q['id']:q for role in ('train','development') for q in read(data/f'STAR_{role}_inputs.json')}
    sk={k:v for role in ('train','development') for k,v in read(data/f'STAR_{role}_keys_LOCAL_ONLY.json').items()}
    states={r['id']:r for r in map(json.loads,(star/'states.jsonl').read_text().splitlines())}
    packed={};inputs=[old/'features.pt',old/'scores.jsonl',star/'features.pt',star/'scores.jsonl',star/'states.jsonl']
    for domain,folder,questions in [('old',old,oq),('STAR',star,sq)]:
        fs=torch.load(folder/'features.pt',weights_only=True,map_location='cpu')
        scores={}
        for line in (folder/'scores.jsonl').read_text().splitlines():
            r=json.loads(line);key=(r['id'],r['subset'],r['shift']);assert key not in scores;scores[key]=r
        for role in ('train','development'):
            ff=[f for f in fs if f['role']==role];n=len(ff)
            assert n=={('old','train'):144,('old','development'):32,('STAR','train'):634,('STAR','development'):171}[domain,role]
            x=[torch.stack([f[k] for f in ff]) for k in FIELDS]
            candidates=torch.zeros(n,16,4,dtype=torch.bool);valid=torch.zeros(n,16,dtype=torch.bool)
            target=torch.zeros(n,16);correct=torch.zeros(n,16);tokens=torch.zeros(n,16)
            mx=[v.clone() for v in x];mx[0].zero_();mx[1].zero_();mx[2].zero_()
            meta=[]
            for j,f in enumerate(ff):
                q=questions[f['id']];gold=sk[f['id']] if domain=='STAR' else q['answer_index'];ps=[]
                choices=f['choices'];nh=int(f['node_mask'].sum());allowed={0,len(choices)-1,choices.index([nh-1]) if nh else 0}
                match=re.search(r'video interval from ([0-9.]+) to ([0-9.]+) seconds',q['question'])
                now=max(float(q['query_time_s']),1e-6)
                mx[0][j,3]=now/120
                if match:
                    a,b=map(float,match.groups());mx[0][j,:3]=torch.tensor([1.,a/now,b/now])
                    assert domain=='STAR' and [a,b]==q['author_interval_s']
                    nodes=states[f['id']]['bank']['nodes'][:nh]
                    allowed.add(choices.index([k for k,node in enumerate(nodes) if node['end']>=a and node['begin']<=b]))
                for i,choice in enumerate(choices):
                    candidates[j,i,choice]=True;valid[j,i]=True;p=[]
                    for shift in (0,2):
                        r=scores[f['id'],i,shift];logits=torch.tensor(r['option_logits']);assert len(logits)==len(q['options'])
                        p.append(float(logits.softmax(-1)[(gold-shift)%len(logits)]));correct[j,i]+=r['selected_original']==gold;tokens[j,i]=r['visual_tokens']
                    ps.append(sum(p)/2)
                target[j,:len(ps)]=torch.tensor(ps)-ps[0]
                meta.append(dict(id=f['id'],source=f['source_id'],domain=domain,role=role,choices=choices,allowed=sorted(allowed),sha=q['media_sha256']))
            packed[domain,role]=dict(x=x+[candidates],mx=mx+[candidates],valid=valid,target=target,correct=correct,tokens=tokens,meta=meta)
    train=[m for (d,r),p in packed.items() if r=='train' for m in p['meta']];dev=[m for (d,r),p in packed.items() if r=='development' for m in p['meta']]
    for field in ('source','sha'):
        assert not ({m[field] for m in train}&{m[field] for m in dev}),field
    return packed,{str(p.relative_to(ROOT)):sha(p) for p in inputs}

def evaluate(head,p,metadata=False):
    with torch.no_grad():pred=head(*(p['mx'] if metadata else p['x']))
    result=dict(huber=float(objective(pred,p['target'],p['valid'],torch.ones(len(pred)))))
    details=[]
    for j,m in enumerate(p['meta']):
        score=pred[j,:len(m['choices'])];allsel=choose(score,m['choices'])
        allowed=m['allowed'];restricted=allowed[choose(score[allowed],[m['choices'][i] for i in allowed])]
        details.append(dict(**m,selected=allsel,restricted=restricted,correct=float(p['correct'][j,allsel]),restricted_correct=float(p['correct'][j,restricted]),
            actual_gain=float(p['target'][j,allsel]),visual_tokens=float(p['tokens'][j,allsel]),selected_nodes=len(m['choices'][allsel])))
    result.update(correct=sum(r['correct'] for r in details),restricted_correct=sum(r['restricted_correct'] for r in details),n=2*len(details),
        null=sum(r['selected']==0 for r in details),mean_nodes=sum(r['selected_nodes'] for r in details)/len(details),
        mean_gain=sum(r['actual_gain'] for r in details)/len(details),mean_visual_tokens=sum(r['visual_tokens'] for r in details)/len(details))
    return result,details

def main():
    assert not OUT.exists();torch.set_num_threads(1);plan=read(HERE/'plan.json');packed,hashes=load();OUT.mkdir()
    save(OUT/'inputs.json',dict(hashes=hashes,plan=plan,plan_sha=sha(HERE/'plan.json'),script_sha=sha(Path(__file__))))
    report={};start=time.monotonic()
    for seed in plan['seeds']:
        for arm in plan['arms']:
            torch.manual_seed(seed);head=Head();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
            gen=torch.Generator().manual_seed(seed);queues={};best=float('inf');curve=[];folder=OUT/f'{arm}-{seed}';folder.mkdir()
            def draw(domain,n):
                key=(domain,'train');total=len(packed[key]['meta']);buf=queues.setdefault(domain,[])
                while len(buf)<n:buf.extend(torch.randperm(total,generator=gen).tolist())
                ix=torch.tensor(buf[:n]);del buf[:n];return packed[key],ix
            for step in range(1,181):
                pieces=[draw('old',16)] if arm=='old' else [draw('old',8),draw('STAR',8)]
                field='mx' if arm=='metadata' else 'x'
                x=[torch.cat([p[field][k][ix] for p,ix in pieces]) for k in range(len(FIELDS)+1)]
                v=torch.cat([p['valid'][ix] for p,ix in pieces]);t=torch.cat([p['target'][ix] for p,ix in pieces])
                opt.zero_grad(set_to_none=True);loss=objective(head(*x),t,v,torch.ones(len(t)));assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
                if step%10==0:
                    metrics={d:evaluate(head,packed[d,'development'],arm=='metadata')[0] for d in ('old','STAR')}
                    devloss=sum(m['huber'] for m in metrics.values())/2
                    curve.append(dict(step=step,online_loss=float(loss.detach()),development_equal_domain_huber=devloss,metrics=metrics))
                    if devloss<best:best=devloss;chosen=step;state=copy.deepcopy(head.state_dict())
            head.load_state_dict(state);head.eval().requires_grad_(False)
            torch.save(dict(state_dict=state,width=3584,arm=arm,seed=seed,updates=180,selected_step=chosen,bank='content70',metadata_only=arm=='metadata'),folder/'head.pt')
            summaries={};details=[]
            for key,p in packed.items():
                summaries['/'.join(key)],dd=evaluate(head,p,arm=='metadata');details+=dd
            save(folder/'curve.json',curve);save(folder/'selections.json',details)
            report[f'{arm}-{seed}']=dict(selected_step=chosen,head_sha=sha(folder/'head.pt'),metrics=summaries)
            save(OUT/'summary.json',report);print(json.dumps(dict(arm=arm,seed=seed,step=chosen,development={k:v for k,v in summaries.items() if 'development' in k})),flush=True)
    save(OUT/'completed.json',dict(fits=9,updates=1620,seconds=time.monotonic()-start,base_updates=0,gpu=0,heldout_opened=False))
if __name__=='__main__':main()
