"""Official answer-content analysis only. Evidence truth remains a separate evaluation."""
import argparse,json
from pathlib import Path
from response_v3 import parse
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',choices=['NExT-QA','StreamingBench'],required=True);a=p.parse_args()
    done=read(a.run/'completed.json');assert done['answers']==360
    qs={q['id']:q for q in read(P/'panel_media_bound.json') if q['dataset']==a.dataset};labels=read(P/'labels_all_LOCAL_ONLY.json')
    rows=[json.loads(x) for x in (a.run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in rows:
        q=qs[r['id']];r.update(parse(r['text'],len(q['options']),r['protocol'],r['shift']))
        # Unknown extraction is not silently converted into a content error.
        r['content_correct']=None if not r['answer_parseable'] else r['original_selected_index']==labels[r['id']]
    arms=('full','fixed12','heuristic11');index={(r['id'],r['arm'],r['protocol'],r['shift']):r for r in rows}
    assert len(index)==len(rows)==360
    assert set(index)=={(q,arm,proto,shift) for q in qs for arm in arms for proto in ('A','B') for shift in (0,2)}
    table={}
    for arm in arms:
        for proto in ('A','B'):
            rr=[r for r in rows if r['arm']==arm and r['protocol']==proto]
            unknown=sum(index[q,arm,proto,0]['content_correct'] is None for q in qs)
            correct=sum(index[q,arm,proto,0]['content_correct'] is True for q in qs)
            table[f'{arm}/{proto}']=dict(original_order_correct=correct,questions=30,
                known_wrong=sum(index[q,arm,proto,0]['content_correct'] is False for q in qs),
                unresolved_content=unknown,correct_count_bounds_before_review=[correct,correct+unknown],
                both_orders_correct=sum(all(index[q,arm,proto,shift]['content_correct'] is True for shift in (0,2)) for q in qs),
                two_orders_correct=sum(r['content_correct'] is True for r in rr),two_orders_answers=60,
                answer_parseable=sum(r['answer_parseable'] for r in rr),answer_conflicts=sum(r['answer_conflict'] for r in rr),
                labeled_evidence=sum(r['explanation_present'] for r in rr),labeled_support=sum(bool(r['support']) for r in rr),
                only_letter_format=sum(r['only_letter_format'] for r in rr),
                generated_tokens=sum(len(r['generated_ids']) for r in rr),generation_cap_reached=sum(len(r['generated_ids'])==(8 if proto=='A' else 128) for r in rr),
                by_type={typ:dict(correct=sum(index[q,arm,proto,0]['content_correct'] is True for q in qs if qs[q]['type']==typ),questions=sum(qs[q]['type']==typ for q in qs)) for typ in sorted({q['type'] for q in qs.values()})})
    gains={}
    for arm in arms:
        gained=[q for q in qs if index[q,arm,'B',0]['content_correct'] is True and index[q,arm,'A',0]['content_correct'] is False]
        lost=[q for q in qs if index[q,arm,'A',0]['content_correct'] is True and index[q,arm,'B',0]['content_correct'] is False]
        unresolved=[q for q in qs if any(index[q,arm,p,0]['content_correct'] is None for p in ('A','B'))]
        gains[arm]=dict(B_vs_A_gained_ids=gained,lost_ids=lost,net_known_only=len(gained)-len(lost),unresolved_pairs=unresolved)
    out=dict(dataset=a.dataset,content=table,protocol_changes=gains,run=done,
        role='descriptive answer reference; not the primary success criterion of evidence evaluation',
        primary_evidence_endpoints=['critical visual facts','support relevance','response to evidence intervention','fabrication under insufficient evidence'],
        sources=len({q['source_id'] for q in qs.values()}),API_evidence_scored=False,
        limitations=['opened development sample','formats not part of accuracy','A/B prompt and token cap differ',
                    'two orders of each question are correlated','explanation presence does not establish truth'])
    (a.run/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:x['original_order_correct'] for k,x in table.items()}))
if __name__=='__main__':main()
