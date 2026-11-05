"""Content and output usability only; does not impersonate a video judge."""
import argparse,collections,json
from pathlib import Path
from response import parse,fact_payload,support_payload
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(x) for x in (a.run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==48 and read(a.run/'completed.json')['answers']==48
    questions={q['id']:q for q in read(P/'profile.json')};labels=read(P/'labels_nextqa.json')
    idx={(r['id'],r['arm'],r['protocol']):r for r in rows}
    assert set(idx)=={(q,arm,proto) for q in questions for arm in ('full','fixed12') for proto in ('A','B')}
    materials={m['material_id']:m for m in read(a.run/'materials.json')}
    summary={};drafts=[]
    for key in [('full','A'),('full','B'),('fixed12','A'),('fixed12','B')]:
        subset=[r for r in rows if (r['arm'],r['protocol'])==key]
        summary['/'.join(key)]=dict(questions=len(subset),content_correct=sum(r['original_selected_index']==labels[r['id']] for r in subset),
            parseable=sum(r['answer_parseable'] for r in subset),explanation_present=sum(r['explanation_present'] for r in subset),
            support_present=sum(bool(r['support']) for r in subset),answer_conflicts=sum(r['answer_conflict'] for r in subset),
            cap_reached=sum(len(r['generated_ids'])==(8 if key[1]=='A' else 128) for r in subset),
            generated_tokens=sum(len(r['generated_ids']) for r in subset),QA_seconds=sum(r['seconds'] for r in subset))
    changes={}
    for arm in ('full','fixed12'):
        gained=[];lost=[]
        for q in questions:
            aok=idx[q,arm,'A']['original_selected_index']==labels[q];bok=idx[q,arm,'B']['original_selected_index']==labels[q]
            if bok and not aok:gained.append(q)
            if aok and not bok:lost.append(q)
        changes[arm]=dict(gained_ids=gained,lost_ids=lost)
    for row in rows:
        if row['protocol']!='B':continue
        q=questions[row['id']];material=materials[row['id']]
        assert material['complete'] and all(0<=f['pts_s']<=q['query_time_s']+1e-6 for f in material['frames'])
        assert len({f['frame_id'] for f in material['frames']})==len(material['frames'])
        first=fact_payload(material,row);assert not {'arm','answer_index','question','options','selected_index'}.intersection(first)
        drafts.append(dict(opaque_id=f'draft-{len(drafts):03d}',fact_request=first,
                           support_request_template=support_payload(q,row,{'pending_fact_verification':True})))
    # These are offline drafts, no actual API/model is selected or contacted.
    (a.run/'judge_request_drafts.json').write_text(json.dumps(drafts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(output_usability=summary,answer_protocol_changes=changes,
        limitations=['12 fixed sources only','explanation presence does not establish truth','API judge unvalidated and unused','A/B prompts and token caps differ'],
        draft_requests=len(drafts)*2,API_calls=0,actual_frame_timing_checks=True)
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
