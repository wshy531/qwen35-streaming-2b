"""Answer-content scoring after inverse option rotation; no format metrics."""
from collections import Counter
from ..long_video_gpu_2b.scoring import scorer
from .controls import ARMS


def build(queries, labels, predictions):
    qs={q['id']:q for q in queries}; gold={r['id']:r for r in labels}
    index={(r['id'],r['arm'],r['shift']):r for r in predictions}
    if (len(qs)!=len(queries) or len(gold)!=len(labels) or set(qs)!=set(gold)
            or len(index)!=len(predictions)
            or set(index)!={(qid,a,s) for qid in qs for a in ARMS for s in range(5)}):
        raise ValueError('complete unique four-arm five-position grid required')
    measure=scorer().measure; details=[]; values={}; choices={}
    for qid,q in qs.items():
        label=gold[qid]; truth=label['correct_index']
        if type(truth) is not int or not 0<=truth<5 or label['correct_letter']!='ABCDE'[truth]:
            raise ValueError('gold index/letter mismatch')
        for arm in ARMS:
            for shift in range(5):
                row=index[qid,arm,shift]; order=[(j+shift)%5 for j in range(5)]
                if (row['status']!='ok' or row['display_to_original']!=order
                        or row['source_path']!=q['source_path'] or row['query_time_s']!=q['query_time_s']):
                    raise ValueError('prediction identity/mapping mismatch')
                m=measure(row['text'],5,(truth-shift)%5)
                displayed=m['content_choice']
                content=order[displayed] if displayed is not None else None
                correct=(content==truth) if content is not None else None
                values[qid,arm,shift]=correct; choices[qid,arm,shift]=content
                details.append(dict(id=qid,arm=arm,shift=shift,source_path=q['source_path'],
                    category=label['question_category'],
                    evidence_duration_s=label['answer_end_time']-label['answer_start_time'],
                    display_choice=displayed,original_choice=content,correct=correct,
                    gold_original=truth,gold_display=(truth-shift)%5,semantic_review='pending'))
    def totals(rows):
        return dict(answers=len(rows),content_correct=sum(r['correct'] is True for r in rows),
            unresolved=sum(r['correct'] is None for r in rows))
    summary={}
    for arm in ARMS:
        rows=[r for r in details if r['arm']==arm]
        total=totals(rows)
        total.update(by_shift={str(s):totals([r for r in rows if r['shift']==s]) for s in range(5)},
            mean_content_accuracy=None if total['unresolved'] else total['content_correct']/len(rows),
            predicted_letters=dict(Counter('ABCDE'[r['display_choice']] for r in rows if r['display_choice'] is not None)),
            by_gold_display={c:totals([r for r in rows if r['gold_display']==i]) for i,c in enumerate('ABCDE')},
            by_category={c:totals([r for r in rows if r['category']==c]) for c in sorted({r['category'] for r in rows})},
            by_video={v:totals([r for r in rows if r['source_path']==v]) for v in sorted({r['source_path'] for r in rows})})
        eligible=[qid for qid in qs if all(choices[qid,arm,s] is not None for s in range(5))]
        total['content_consistency']=dict(evaluable_questions=len(eligible),
            all_five_same=sum(len({choices[qid,arm,s] for s in range(5)})==1 for qid in eligible),
            changed_content_ids=[qid for qid in eligible if len({choices[qid,arm,s] for s in range(5)})>1])
        summary[arm]=total
    paired={}
    for a,b in (('R','B'),('F','B'),('B','T')):
        common=[(q,s) for q in qs for s in range(5) if values[q,a,s] is not None and values[q,b,s] is not None]
        gains=[dict(id=q,shift=s) for q,s in common if values[q,a,s] and not values[q,b,s]]
        losses=[dict(id=q,shift=s) for q,s in common if values[q,b,s] and not values[q,a,s]]
        per_question=[]
        for q in qs:
            if all(values[q,a,s] is not None and values[q,b,s] is not None for s in range(5)):
                delta=sum(int(values[q,a,s])-int(values[q,b,s]) for s in range(5))/5
                per_question.append(dict(id=q,mean_accuracy_delta=delta))
        paired[a+'_vs_'+b]=dict(evaluable_pairs=len(common),corrections=gains,losses=losses,
            net=len(gains)-len(losses),per_question=per_question)
    return dict(totals=summary,paired=paired,independent_questions=len(qs),
        independent_videos=len({q['source_path'] for q in queries}),
        interpretation='five rotations are repeated measures, not independent questions',
        human_semantic_review='pending'),details
