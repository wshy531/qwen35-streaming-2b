"""Separate score-based primary outcomes, generated audit, and diagnostic pairs."""
import argparse
from collections import defaultdict
from common import *
sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
from response_v4 import parse

def main(run):
    complete=read(run/'completed.json');assert complete['answers']==3456
    qs={q['id']:q for q in read(PREP/'evaluation.json')};gold=read(PREP/'evaluation_keys_LOCAL_ONLY.json')
    rows=[json.loads(s) for s in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    keys={(r['id'],r['mode'],r['route'],r['shift']) for r in rows}
    assert len(keys)==len(rows)==3456
    for r in rows:r.update(correct=r['selected_original']==gold[r['id']],gold=gold[r['id']],type=qs[r['id']]['type'])
    tables=[];paired=[]
    for suite in ('SB150','S-EMBER-early-candidate','D18'):
        for route in ('recent8','LS'):
            for mode in ('full','fixed','content'):
                rs=[r for r in rows if (r['suite'],r['route'],r['mode'])==(suite,route,mode)]
                perq=defaultdict(list)
                for r in rs:perq[r['id']].append(r)
                tables.append(dict(suite=suite,route=route,mode=mode,n_answers=len(rs),n_questions=len(perq),
                    correct=sum(r['correct'] for r in rs),original_order_correct=sum(r['correct'] for r in rs if r['shift']==0),
                    both_orders_correct=sum(all(r['correct'] for r in group) for group in perq.values()),
                    mean_visual_tokens=sum(r['visual_tokens'] for r in rs)/len(rs)))
                if mode!='full':
                    baseline={(r['id'],r['shift']):r for r in rows if (r['suite'],r['route'],r['mode'])==(suite,route,'full')}
                    gain=[(r['id'],r['shift']) for r in rs if r['correct'] and not baseline[r['id'],r['shift']]['correct']]
                    loss=[(r['id'],r['shift']) for r in rs if not r['correct'] and baseline[r['id'],r['shift']]['correct']]
                    paired.append(dict(suite=suite,route=route,mode=mode,gain=gain,loss=loss))
    audits=[json.loads(s) for s in (run/'audits.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(audits)==12
    for a in audits:
        for name in ('native','stream','single_video_native'):
            p=parse(a[name+'_text'],len(qs[a['id']]['options']),'B');selected=None if p['answer_conflict'] else p['original_selected_index']
            a[name+'_correct']=selected==gold[a['id']];a[name+'_selected']=selected
    output=dict(tables=tables,paired=paired,audits=audits,per_question=rows,
        evidence='real 7B FP32; untrained fixed/content pruning, normalized option scoring',
        caveats=['Opened development sources, not unseen generalization','Two option orders are correlated',
                 'Same pair capacities, half token budget after pruning; not pure token-location intervention',
                 'Shared prefix encoding is not standalone cost','Native audit is a length-stratified 12-case diagnosis, capped at 16384 tokens'])
    save(run/'analysis.json',output);print(json.dumps(dict(tables=tables,audit_passed=sum(a['passed'] for a in audits)),ensure_ascii=False,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);main(ap.parse_args().run)
