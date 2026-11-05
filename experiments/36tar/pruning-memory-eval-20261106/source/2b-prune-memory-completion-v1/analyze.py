"""Content correctness and same-time current/old pairs, on complete runs only."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
from response_v4 import parse
PREP=ROOT/'runs/prune_memory_completion_prepare_20261105_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main(panel,out):
    plan=read(PREP/'plan.json')[panel]
    done=read(out/'completed.json')
    assert done['status']=='completed' and done['base_updates']==done['scorer_updates']==0
    for k in ('answers','reference_matches','pairs','vision','sources'):assert done[k]==plan[k]
    qs=read(PREP/f'{panel}_questions.json');tasks={q['id']:q for q in qs}
    if panel=='stream':labels=read(ROOT/'runs/evidence_scale_prepare_20261102_01/labels_LOCAL_ONLY.json')
    else:labels={k:v['correct_index'] for k,v in read(ROOT/'runs/midprune_memory_prepare_20261102_01/labels_LOCAL_ONLY.json').items()}
    rows={}
    for line in (out/'answers.jsonl').open(encoding='utf-8'):
        r=json.loads(line);key=r['id'],r['arm']
        assert key not in rows and r['id'] in tasks and r['branch_released'] and r['read_state_unchanged']
        assert all(n['end']<=tasks[r['id']]['query_time_s']+1e-6 for n in r['nodes'])
        r.update(parse(r['text'],len(tasks[r['id']]['options']),'B'))
        choice=None if r['answer_conflict'] else r['original_selected_index']
        r['correct']=None if choice is None else choice==labels[r['id']]
        rows[key]=r
    arms=sorted({a for _,a in rows})
    assert len(rows)==plan['answers'] and set(rows)=={(q['id'],a) for q in qs for a in arms}
    def group(items):
        result={}
        for arm in arms:
            r=[rows[q['id'],arm] for q in items]
            result[arm]=dict(correct=sum(x['correct'] is True for x in r),unresolved=sum(x['correct'] is None for x in r),
                total=len(r),mean_visual_tokens=sum(x['visual_tokens'] for x in r)/len(r))
        return result
    groups={'all':group(qs)}
    if panel=='paired':
        for role in ('current','old'):groups[role]=group([q for q in qs if q['role']==role])
        both={}
        for arm in arms:
            count=0
            for old in (q for q in qs if q['role']=='old'):
                current,=[q for q in qs if q['case_id']==old['case_id'] and q['role']=='current' and q['stage']=='late']
                assert current['query_time_s']==old['query_time_s']
                count+=rows[old['id'],arm]['correct'] is True and rows[current['id'],arm]['correct'] is True
            both[arm]=dict(correct=count,total=6)
        groups['same_time_both']=both
    else:
        for kind in sorted({q['type'] for q in qs}):groups['type/'+kind]=group([q for q in qs if q['type']==kind])
        for length in sorted({q['length_bin'] for q in qs}):groups['length/'+length]=group([q for q in qs if q['length_bin']==length])
    paired={}
    for arm in arms:
        if not arm.startswith('guarded'):continue
        paired[arm]={}
        for base in arms:
            pairs=[(rows[q['id'],arm]['correct'],rows[q['id'],base]['correct']) for q in qs]
            paired[arm][base]=dict(gain=pairs.count((True,False)),loss=pairs.count((False,True)),
                unresolved=sum(None in p for p in pairs))
    result=dict(panel=panel,groups=groups,guarded_paired=paired,
        note='Unresolved answers require separate content review. They are not silently called wrong due to format.')
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=('stream','paired'),required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.panel,a.out)
