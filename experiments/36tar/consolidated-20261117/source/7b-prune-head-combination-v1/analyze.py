"""Fixed-combination results, paired quality and measured QA component costs."""
import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_prune_head_prepare_20261111_01'
OUT=ROOT/'runs/7b-prune-head-223298'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
done=read(OUT/'completed.json')
assert all(done[k]==v for k,v in dict(questions=210,classifications=2940,references=1860,state_checks=210,representations=420,base_updates=0,head_updates=0).items())
for rel,h in read(PREP/'lock.json').items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
rows=lines(OUT/'answers.jsonl');routes=lines(OUT/'routes.jsonl')
by={(r['id'],r['mode'],r['route'],r['shift']):r for r in rows};assert len(by)==len(rows)==2940
rb={(r['id'],r['mode']):r for r in routes};assert len(rb)==len(routes)==420
qs={q['id']:q for q in read(PREP/'questions.json')}
gold=read(ROOT/'runs/7b_method_transfer_prepare_20261110_01/evaluation_keys_LOCAL_ONLY.json')
gold.update(read(ROOT/'runs/7b_selector_short_prepare_20261110_01/keys_LOCAL_ONLY.json'))
arms=[('full','recent8'),('full','LS'),('full','head'),('content','recent8'),('content','LS'),('content','head'),('content','same_count_earliest')]
for i,q in qs.items():
    for mode,route in arms:
        for shift in (0,2):
            r=by[i,mode,route,shift]
            assert r['suite']==q['suite'] and r['source_id']==q['source_id']
            r['correct']=r['selected_original']==gold[i]
for r in read(PREP/'references.json'):
    actual=by[r['id'],r['mode'],r['route'],r['shift']]
    assert actual['selected_original']==r['selected_original'] and actual['visual_tokens']==r['visual_tokens']
    assert np.allclose(actual['option_logits'],r['option_logits'],atol=1e-5,rtol=1e-5)
for i,choice in read(PREP/'full_selections.json').items():assert rb[i,'full']['choice']==choice
tables=[];pairs={};selection={};types=[]
for suite in ('SB150','short60'):
    ids=[i for i,q in qs.items() if q['suite']==suite]
    for mode,route in arms:
        a=[by[i,mode,route,s] for i in ids for s in (0,2)]
        tables.append(dict(suite=suite,mode=mode,route=route,correct=sum(r['correct'] for r in a),n=len(a),
            both_orders_correct=sum(all(by[i,mode,route,s]['correct'] for s in (0,2)) for i in ids),
            mean_visual_tokens=float(np.mean([r['visual_tokens'] for r in a])),
            mean_read_seconds=float(np.mean([r['read_seconds'] for r in a])),
            mean_selection_seconds=float(np.mean([r['selection_seconds'] for r in a])),
            mean_QA_component_seconds=float(np.mean([r['read_seconds']+r['selection_seconds'] for r in a]))))
        for kind in sorted({qs[i].get('type','unlabeled') for i in ids}):
            subset=[r for r in a if qs[r['id']].get('type','unlabeled')==kind]
            types.append(dict(suite=suite,type=kind,mode=mode,route=route,correct=sum(r['correct'] for r in subset),n=len(subset)))
    selection[suite]={}
    for mode in ('full','content'):
        a=[rb[i,mode] for i in ids]
        selection[suite][mode]=dict(node_counts=dict(Counter(len(r['choice']) for r in a)),
            subsets=dict(Counter(str(r['choice']) for r in a)),mean_seconds=float(np.mean([r['selection_seconds'] for r in a])),
            norm_means={k:float(np.mean([r['norms'][k] for r in a])) for k in a[0]['norms']})
    selection[suite]['route_changes']=sum(rb[i,'full']['choice']!=rb[i,'content']['choice'] for i in ids)
    for baseline in [('full','recent8'),('full','LS'),('full','head'),('content','LS'),('content','recent8'),('content','same_count_earliest')]:
        gains=losses=0;sources=defaultdict(list);changes=[]
        for i in ids:
            d=0
            for s in (0,2):
                a=by[i,'content','head',s];b=by[i,*baseline,s]
                gains+=a['correct'] and not b['correct'];losses+=b['correct'] and not a['correct'];d+=int(a['correct'])-int(b['correct'])
            sources[qs[i]['source_id']].append(d)
            if d:changes.append(dict(id=i,question=qs[i]['question'],type=qs[i].get('type'),delta=d))
        z=np.array([(sum(v),2*len(v)) for v in sources.values()]);rng=np.random.default_rng(7111)
        sample=z[rng.integers(len(z),size=(10000,len(z)))].sum(1)
        pairs[suite+' vs '+'_'.join(baseline)]=dict(gains=int(gains),losses=int(losses),net=int(gains-losses),
            source_bootstrap_95=np.quantile(sample[:,0]/sample[:,1],[.025,.975]).tolist(),changed_questions=changes)
result=dict(tables=tables,selection=selection,paired=pairs,types=types,
    checks=dict(unique_rows=2940,reference_rows=1860,full_routes=210,checkpoint_unchanged=True),
    limits=['Two permutations are correlated; 210 unique questions.','QA costs include selection once per hypothetical answer, not independent end-to-end runs.','LS is bounded saved history, not complete original history.','No S-EMBER or D18.'])
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for row in tables:print(json.dumps(row))
print('SELECTION',json.dumps(selection))
for k,v in pairs.items():print(k,json.dumps({x:y for x,y in v.items() if x!='changed_questions'}))
