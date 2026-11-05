"""Offline option-content accounting. Explanations require separate evidence judgement."""
import argparse,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/midprune_memory_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def analyze(rows,labels):
    methods={}
    for r in rows:
        key=r['view']+'/'+r['policy'];d=methods.setdefault(key,{})
        assert r['id'] not in d and r['id'] in labels
        assert r['read_state_unchanged'] and r['branch_released']
        assert r['bank_status']['tokens']<=r['bank_status']['visual_budget']
        d[r['id']]=dict(correct=r['original_selected_index']==labels[r['id']]['correct_index'],row=r)
    assert len(methods)==19 and all(set(x)==set(labels) for x in methods.values())
    summaries={}
    for name,data in methods.items():
        out={}
        for role in ('current','old'):
            ids=[i for i,l in labels.items() if l['role']==role]
            out[role]=dict(questions=len(ids),option_content_correct=sum(data[i]['correct'] for i in ids),
                parseable=sum(data[i]['row']['answer_parseable'] for i in ids))
        out['late_both_correct']=sum(all(data[f'{case}:late:{role}']['correct'] for role in ('current','old')) for case in sorted({l['case_id'] for l in labels.values()}))
        out['late_pairs']=6
        out['maximum_bank_tokens']=max(v['row']['bank_status']['tokens'] for v in data.values())
        base=methods['full/'+name.split('/')[1].replace('M0_same8','M0')]
        out['versus_full_same_policy']=dict(gained=[i for i in data if data[i]['correct'] and not base[i]['correct']],lost=[i for i in data if base[i]['correct'] and not data[i]['correct']])
        summaries[name]=out
    return dict(methods=summaries,limitations=['18 familiar single-assistant development questions, not an independent generalization test',
        'Option content correctness is not evidence-supported explanation correctness; no new judge results',
        'Old evidence necessity is not established for all cases; preserve original label caveats',
        'Shared quality run timing is not standalone or edge-device cost',
        'Pruning can change event boundaries; same temporal coverage control applies only to M0_same8'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    done=read(a.run/'completed.json');assert done['status']=='complete' and done['answers']==342
    rows=[json.loads(s) for s in (a.run/'answers.jsonl').read_text().splitlines()]
    result=analyze(rows,read(P/'labels_LOCAL_ONLY.json'));result['run']=done
    with (a.run/'analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
