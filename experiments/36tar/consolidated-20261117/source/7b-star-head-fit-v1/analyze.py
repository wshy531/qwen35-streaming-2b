from fit import *
from collections import Counter,defaultdict
import numpy as np
torch.set_num_threads(1)
p,_=load();summary=read(OUT/'summary.json');analysis={};signals={}
for key,pack in p.items():
    groups=defaultdict(list)
    for j,m in enumerate(pack['meta']):
        for i,choice in enumerate(m['choices']):groups[len(choice)].append(float(pack['target'][j,i]))
    signals['/'.join(key)]={str(k):dict(n=len(v),mean_gain=sum(v)/len(v)) for k,v in groups.items()}
for name,result in summary.items():
    rows=read(OUT/name/'selections.json');item={}
    for domain in ('old','STAR'):
        rr=[r for r in rows if r['domain']==domain and r['role']=='development']
        item[domain]=dict(node_count_histogram=dict(Counter(r['selected_nodes'] for r in rr)),
            no_history=sum(len(r['choices'])==1 for r in rr),
            voluntary_null=sum(r['selected']==0 and len(r['choices'])>1 for r in rr),
            all_history=sum(r['selected']==len(r['choices'])-1 for r in rr),
            final_node=sum(r['choices'][r['selected']]==[max(max(c) for c in r['choices'] if c)] for r in rr if len(r['choices'])>1))
    curve=read(OUT/name/'curve.json');item['loss']=dict(first=curve[0]['development_equal_domain_huber'],chosen=next(r['development_equal_domain_huber'] for r in curve if r['step']==result['selected_step']),last=curve[-1]['development_equal_domain_huber'])
    analysis[name]=item
paired={}
for seed in (20261104,7113,17):
    old={r['id']:r for r in read(OUT/f'old-{seed}'/'selections.json')}
    mixed=read(OUT/f'mixed-{seed}'/'selections.json')
    for domain in ('old','STAR'):
        rr=[r for r in mixed if r['domain']==domain and r['role']=='development'];src=defaultdict(lambda:[0,0]);gain=loss=0
        for r in rr:
            b=old[r['id']];d=r['correct']-b['correct'];gain+=max(d,0);loss+=max(-d,0);src[r['source']][0]+=d;src[r['source']][1]+=2
        z=np.array(list(src.values()));rng=np.random.default_rng(7113);v=z[rng.integers(len(z),size=(3000,len(z)))].sum(1)
        paired[f'{domain}-{seed}']=dict(gain=gain,loss=loss,unit='net two-permutation per-question change, not answer-wise transitions',source_bootstrap95=np.quantile(v[:,0]/v[:,1],[.025,.975]).tolist())
save(OUT/'behavior_analysis.json',dict(behavior=analysis,paired=paired,signal_by_cardinality=signals))
for name,x in analysis.items():print(name,json.dumps(x['STAR']),json.dumps(x['loss']))
# Frozen former repaired head: reference only; its checkpoint used a different development rule.
path=ROOT/'runs/7b_head_stability_prepare_20261112_01/head-control.pt'
head=Head();ck=torch.load(path,map_location='cpu',weights_only=True);head.load_state_dict(ck['state_dict'])
ref={k:evaluate(head,p[k])[0] for k in [('old','development'),('STAR','development')]}
save(OUT/'previous_head_reference.json',dict(path=str(path.relative_to(ROOT)),sha=sha(path),metrics={'/'.join(k):v for k,v in ref.items()},different_development_rule=True))
print('former_head',ref)
