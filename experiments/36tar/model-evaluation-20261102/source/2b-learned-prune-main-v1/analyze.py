"""Content correctness first, format separately; video-level paired uncertainty."""
import argparse,collections,json
from protocol import PREP,ARMS,read

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=__import__('pathlib').Path,required=True);a=parser.parse_args()
    done=read(a.run/'completed.json');assert done['status']=='complete'
    questions={q['id']:q for q in read(PREP/'eval.json')}
    rows=[json.loads(line) for line in (a.run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    indexed={(r['id'],r['arm'],r['shift']):r for r in rows}
    expected={(q,arm,shift) for q in questions for arm in ARMS for shift in [0,2]}
    assert len(indexed)==len(rows)==7824 and set(indexed)==expected
    for r in rows:
        q=questions[r['id']];assert r['role']==q['role'] and r['video_id']==q['video_id']
        assert r['correct']==(r['original_selected_index']==q['answer_index'])
    result={}
    import numpy as np
    for role in ['development','confirmation']:
        ids=[q for q in questions if questions[q]['role']==role];n=len(ids)
        table={}
        for arm in ARMS:
            rr=[indexed[q,arm,s] for q in ids for s in [0,2]]
            table[arm]=dict(primary_correct=sum(indexed[q,arm,0]['correct'] for q in ids),primary_questions=n,
                two_rotations_correct=sum(r['correct'] for r in rr),two_rotations_answers=2*n,
                both_rotations_correct=sum(all(indexed[q,arm,s]['correct'] for s in [0,2]) for q in ids),
                same_answer_content=sum(indexed[q,arm,0]['original_selected_index'] is not None and
                    indexed[q,arm,0]['original_selected_index']==indexed[q,arm,2]['original_selected_index'] for q in ids),
                only_letter_format=sum(r['format_only_letter'] for r in rr),
                by_type={typ:dict(correct=sum(indexed[q,arm,0]['correct'] for q in ids if questions[q]['type']==typ),
                                 questions=sum(questions[q]['type']==typ for q in ids))
                         for typ in sorted({questions[q]['type'] for q in ids})})
        comparisons={};videos=sorted({questions[q]['video_id'] for q in ids})
        for reference in ARMS:
            if reference=='trained':continue
            gain=sum(indexed[q,'trained',0]['correct'] and not indexed[q,reference,0]['correct'] for q in ids)
            loss=sum(not indexed[q,'trained',0]['correct'] and indexed[q,reference,0]['correct'] for q in ids)
            counts=np.array([sum(questions[q]['video_id']==v for q in ids) for v in videos])
            delta=np.array([sum(int(indexed[q,'trained',0]['correct'])-int(indexed[q,reference,0]['correct'])
                            for q in ids if questions[q]['video_id']==v) for v in videos])
            rng=np.random.default_rng(20261101);sample=rng.integers(0,len(videos),size=(2000,len(videos)))
            boot=100*delta[sample].sum(1)/counts[sample].sum(1)
            comparisons[reference]=dict(gained=gain,lost=loss,net=gain-loss,
                primary_difference_pp=100*(gain-loss)/n,video_bootstrap_95_pp=np.quantile(boot,[.025,.975]).tolist())
        result[role]=dict(methods=table,trained_vs=comparisons,video_files=len(videos),
                          capture_parent_independence='unknown; exploratory intervals only')
    output=dict(status='complete',quality=result,run=done,
                cost_limit='shared decode and visual reuse across questions; no independent end-to-end speed claim',
                rotation_limit='two arrangements of the same questions, not independent added questions')
    (a.run/'analysis.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({role:{arm:d['primary_correct'] for arm,d in result[role]['methods'].items()} for role in result}))

if __name__=='__main__':main()
