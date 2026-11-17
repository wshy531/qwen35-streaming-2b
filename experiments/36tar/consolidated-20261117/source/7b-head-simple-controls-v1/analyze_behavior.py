"""Post-hoc attribution of frozen H1 results; no fitting or model selection."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/7b-head-simple-223656'

def read(p): return json.loads(p.read_text(encoding='utf8'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]

def main():
    tasks = {t['question']['id']: t for t in read(ROOT/'runs/7b_head_simple_prepare_20261112_01/tasks.json')}
    gold = read(ROOT/'runs/7b_method_transfer_prepare_20261110_01/evaluation_keys_LOCAL_ONLY.json')
    gold.update(read(ROOT/'runs/7b_selector_short_prepare_20261110_01/keys_LOCAL_ONLY.json'))
    data = {}
    for p in [ROOT/'runs/7b-pruned-head-adapt-223422/final/answers.jsonl',
              ROOT/'runs/7b-head-stability-223607/final/answers.jsonl', OUT/'answers.jsonl']:
        for r in lines(p): data[r['id'],r['route'],r['shift']] = r
    routes = {r['id']: r for r in lines(ROOT/'runs/7b-head-stability-223607/final/routes.jsonl') if r.get('kind')=='repair'}
    result = {}
    for suite in ('SB150', 'short60'):
        ids = [i for i,t in tasks.items() if t['question']['suite']==suite]
        info = {'questions':len(ids),'sources':len({tasks[i]['question']['source_id'] for i in ids}), 'heads':{},'types':{}}
        for head, field, same in [('control','control_choice','control_last'),('small_step','small_choice','small_last')]:
            cs = [routes[i][field] for i in ids]
            item = dict(selected_counts=dict(collections.Counter(len(c) for c in cs)),
                        same_subset_as_fixed_last=sum(routes[i][field]==tasks[i]['methods']['last'] for i in ids),
                        same_subset_as_same_count_last=sum(routes[i][field]==tasks[i]['methods'][same] for i in ids), comparisons={})
            for other in ('last',same,'recent8','LS'):
                gains,losses,ansdiff,details = 0,0,0,[]
                for i in ids:
                    g=l=ad=0
                    for shift in (0,2):
                        a,b=data[i,head,shift],data[i,other,shift]
                        ac,bc=a['selected_original']==gold[i], b['selected_original']==gold[i]
                        g+=int(ac and not bc); l+=int(bc and not ac)
                        ad+=int(a['selected_original']!=b['selected_original'])
                    gains+=g;losses+=l;ansdiff+=ad
                    if g or l:
                        q=tasks[i]['question']
                        details.append(dict(id=i,source=q['source_id'],question=q['question'],type=q['type'],gains=g,losses=l,
                                            chosen=routes[i][field],gold=gold[i]))
                item['comparisons'][other]=dict(gains=gains,losses=losses,answer_content_differences=ansdiff,changed_correctness=details)
            info['heads'][head]=item
        for kind in sorted({tasks[i]['question']['type'] for i in ids}):
            ii=[i for i in ids if tasks[i]['question']['type']==kind]
            info['types'][kind]={'n':2*len(ii), **{m:sum(data[i,m,s]['selected_original']==gold[i] for i in ii for s in (0,2))
                for m in ('recent8','LS','control','small_step','first','last')}}
        result[suite]=info
    (OUT/'behavior_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({s:{'questions':r['questions'],'sources':r['sources'], 'heads':{h:{k:v for k,v in x.items() if k!='comparisons'} for h,x in r['heads'].items()},'types':r['types']} for s,r in result.items()}))

if __name__=='__main__': main()
