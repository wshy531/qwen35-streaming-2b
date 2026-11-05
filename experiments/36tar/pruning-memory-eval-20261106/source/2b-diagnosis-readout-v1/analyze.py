"""Score all frozen diagnostic conditions; never filter questions by model success."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PREP=ROOT/'runs/diagnosis_readout_prepare_20261105_01'
sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
from response_v4 import parse
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main(args):
    done=read(args.run/'completed.json');assert done['status']=='completed' and done['answers']==88
    qs=read(PREP/'questions.json');labels=read(PREP/'labels_LOCAL_ONLY.json')
    answers=[json.loads(line) for line in (args.run/'answers.jsonl').read_text('utf-8').splitlines()]
    indexed={}
    for a in answers:
        key=a['id'],a['arm'];assert key not in indexed
        q=next(q for q in qs if q['id']==a['id'])
        p=parse(a['text'],len(q['options']),'B')
        choice=None if p['answer_conflict'] else p['original_selected_index']
        indexed[key]=dict(a,correct=None if choice is None else choice==labels[a['id']])
    assert set(indexed)=={(q['id'],arm) for q in qs for arm in ('recent','early','combined','text')}
    table={}
    for arm in ('recent','early','combined','text'):
        table[arm]={}
        for role in ('current','history','all'):
            rows=[indexed[q['id'],arm] for q in qs if role=='all' or q['type']==role]
            table[arm][role]=dict(correct=sum(r['correct'] is True for r in rows),
                unresolved=sum(r['correct'] is None for r in rows),total=len(rows),
                truncated=sum(r['reason']!='eos' for r in rows))
        pairs={q['source_id'] for q in qs}
        table[arm]['both_correct']=sum(all(indexed[q['id'],arm]['correct'] is True
            for q in qs if q['source_id']==source) for source in pairs)
    comparisons={}
    for role in ('current','history'):
        subset=[q for q in qs if q['type']==role]
        comparisons[role]=dict(
            combined_gain=[q['id'] for q in subset if indexed[q['id'],'combined']['correct'] is True and indexed[q['id'],'recent']['correct'] is False],
            combined_loss=[q['id'] for q in subset if indexed[q['id'],'combined']['correct'] is False and indexed[q['id'],'recent']['correct'] is True],
            early_unreadable=[q['id'] for q in subset if indexed[q['id'],'early']['correct'] is False])
    result=dict(table=table,comparisons=comparisons,
        per_question=[dict(id=q['id'],role=q['type'],source_id=q['source_id'],results={arm:indexed[q['id'],arm]['correct'] for arm in table}) for q in qs],
        limitations=['11 correlated source pairs; single assistant annotations',
         'Human-selected uncompressed evidence; no automatic memory retrieval tested',
         'Option content correctness; explanations not externally graded',
         'All draft questions retained, including failures and prior-sensitive cases'])
    out=args.out;assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(table,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
