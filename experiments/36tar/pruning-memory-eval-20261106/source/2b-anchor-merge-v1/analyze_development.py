import argparse
from collections import Counter
from development_worker import R,P,read,save
def main(out):
    import json
    assert read(out/'completed.json')['status']=='complete'
    rows=[json.loads(x) for x in (out/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    qs={q['id']:q for q in read(P/'questions.json')};arms=read(P/'protocol.json')['arms']
    expected={(q,a,s) for q in qs for a in arms for s in (0,2)}
    keyed={(x['id'],x['arm'],x['shift']):x for x in rows};assert len(rows)==len(keyed)==len(expected)==2760 and set(keyed)==expected
    for key,row in keyed.items():
        original=None if row['selected_index'] is None else (row['selected_index']+row['shift'])%5
        assert row['correct']==(original==qs[row['id']]['answer_index'])
    summaries={}
    for shift in (0,2):
        scores={a:dict(correct=sum(keyed[q,a,shift]['correct'] for q in qs),total=345,unparsed=sum(keyed[q,a,shift]['selected_index'] is None for q in qs)) for a in arms}
        paired={}
        for base in arms[:-1]:
            paired[base]=dict(gained=[q for q in qs if keyed[q,'anchor_merge11',shift]['correct'] and not keyed[q,base,shift]['correct']],lost=[q for q in qs if keyed[q,base,shift]['correct'] and not keyed[q,'anchor_merge11',shift]['correct']])
        summaries[str(shift)]=dict(scores=scores,paired=paired)
    save(out/'analysis.json',dict(summaries=summaries,questions=345,sources=40,rotations_not_independent_questions=True,development_only=True))
    print({s:x['scores'] for s,x in summaries.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=__import__('pathlib').Path);main(p.parse_args().out)
