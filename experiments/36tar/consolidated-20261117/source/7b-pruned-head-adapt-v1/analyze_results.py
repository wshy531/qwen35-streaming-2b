"""Local post-freeze scoring only. Evaluation gold never enters the GPU job."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_pruned_head_adapt_prepare_20261111_01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]

def analyze(out,partial=False):
    questions={q['id']:q for q in read(PREP/'questions.json')}
    gold=read(ROOT/'runs/7b_method_transfer_prepare_20261110_01/evaluation_keys_LOCAL_ONLY.json')
    gold.update(read(ROOT/'runs/7b_selector_short_prepare_20261110_01/keys_LOCAL_ONLY.json'))
    arms=['recent8','LS','head','adapted','adapted_same_count_earliest']
    rows=lines(out/'answers.jsonl');by={(r['id'],r['route'],r['shift']):r for r in rows}
    assert len(by)==len(rows)
    complete=[i for i in questions if all((i,a,s) in by for a in arms for s in (0,2))]
    if not partial:
        done=read(out/'completed.json')
        for k,v in dict(train_questions=176,eval_questions=210,references=1260,route_checks=210,state_checks=210,head_updates=180,base_updates=0,generated_tokens=0).items():assert done[k]==v
        assert len(complete)==210 and len(rows)==2100
        assert done['head_sha']==hashlib.sha256((out/'head-visible.pt').read_bytes()).hexdigest()
    for r in read(PREP/'references.json'):
        k=r['id'],r['route'],r['shift']
        if k not in by:continue
        actual=by[k]
        assert actual['selected_original']==r['selected_original'] and actual['visual_tokens']==r['visual_tokens']
        assert np.allclose(actual['option_logits'],r['option_logits'],rtol=1e-5,atol=1e-5)
    for r in rows:
        assert r['suite']==questions[r['id']]['suite'] and r['source_id']==questions[r['id']]['source_id']
        r['correct']=r['selected_original']==gold[r['id']]
    routes={r['id']:r for r in lines(out/'routes.jsonl')}
    tables=[];paired={};selection={};type_rows=[]
    for suite in ('SB150','short60'):
        ids=[i for i in complete if questions[i]['suite']==suite]
        if not ids:continue
        for arm in arms:
            rs=[by[i,arm,s] for i in ids for s in (0,2)]
            tables.append(dict(suite=suite,method=arm,correct=sum(r['correct'] for r in rs),n=len(rs),questions=len(ids),
                both_orders_correct=sum(all(by[i,arm,s]['correct'] for s in (0,2)) for i in ids),
                mean_visual_tokens=float(np.mean([r['visual_tokens'] for r in rs]))))
            for kind in sorted({questions[i].get('type','unlabeled') for i in ids}):
                sub=[r for r in rs if questions[r['id']].get('type','unlabeled')==kind]
                type_rows.append(dict(suite=suite,method=arm,type=kind,n=len(sub),correct=sum(r['correct'] for r in sub)))
        selection[suite]=dict(old_node_counts=dict(Counter(len(routes[i]['old_choice']) for i in ids)),
            new_node_counts=dict(Counter(len(routes[i]['choice']) for i in ids)),
            changed=sum(routes[i]['old_choice']!=routes[i]['choice'] for i in ids))
        for base in [a for a in arms if a!='adapted']:
            gains=losses=0;clusters=defaultdict(list);changed=[]
            for i in ids:
                d=0
                for s in (0,2):
                    a,b=by[i,'adapted',s]['correct'],by[i,base,s]['correct']
                    gains+=a and not b;losses+=b and not a;d+=int(a)-int(b)
                clusters[questions[i]['source_id']].append(d)
                if d:changed.append(dict(id=i,delta=d,question=questions[i]['question']))
            z=np.array([(sum(v),len(v)*2) for v in clusters.values()]);rng=np.random.default_rng(711111)
            boot=z[rng.integers(len(z),size=(10000,len(z)))].sum(1)
            paired[suite+' vs '+base]=dict(gains=int(gains),losses=int(losses),net=int(gains-losses),
                source_bootstrap_95=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist(),changed_questions=changed)
    result=dict(partial=partial,complete_questions=len(complete),tables=tables,paired=paired,selection=selection,types=type_rows,
        limitations=['Opened development/regression data, not unseen test.','Two option orders are correlated.',
        'Option classification, not free-form evidence evaluation.','LS is a bounded bank, not full original history.',
        'No independent end-to-end or edge-device timing.'])
    (out/('partial_analysis.json' if partial else 'analysis.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for row in tables:print(json.dumps(row))
    for name,row in paired.items():print(name,json.dumps({k:v for k,v in row.items() if k!='changed_questions'}))
    return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--partial',action='store_true')
    args=a.parse_args();analyze(args.out,args.partial)
