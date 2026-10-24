"""Frozen strict scoring. All missing/error answers stay in each original denominator."""
from collections import defaultdict
import argparse
from pathlib import Path

from common import OUT, read, save, sha, ARMS
from score_memory_eval import parse_choice


def analyze(result, destination):
    rows={q['id']:q for q in read(OUT/'inputs/questions.json')}
    labels={q['id']:q for q in read(OUT/'labels_LOCAL_ONLY.json')}
    predictions={}
    path=result/'answers.jsonl'
    if path.is_file():
        import json
        for line in path.open(encoding='utf-8'):
            p=json.loads(line);key=p['id'],p['arm']
            assert key[0] in rows and key[1] in ARMS and key not in predictions
            predictions[key]=p
    details=[]
    for uid,q in rows.items():
        for arm in ARMS:
            p=predictions.get((uid,arm),{})
            choice=parse_choice(p.get('text'),len(q['options'])) if p.get('status')=='ok' else None
            details.append(dict(id=uid,group=q['group'],cohort=q['cohort'],arm=arm,text=p.get('text'),
                truth=labels[uid]['answer_index'],prediction=choice,correct=choice==labels[uid]['answer_index'],
                status='missing' if not p else p.get('status') if p.get('status')!='ok' else 'format_error' if choice is None else 'ok',
                generated_ids=p.get('generated_ids',[]),input_tokens=p.get('input_tokens'),visual_tokens=p.get('visual_tokens')))
    lookup={(p['id'],p['arm']):p for p in details}
    subsets={}
    for cohort in ('natural171','T98'):
        subsets[cohort]=[uid for uid,q in rows.items() if q['cohort']==cohort]
        for stratum in ('F','T','C'):
            subsets[cohort+'_'+stratum]=[uid for uid in subsets[cohort] if labels[uid]['stratum']==stratum]
    subsets['primary269']=[uid for uid,q in rows.items() if q['cohort']!='stream34']
    subsets['stream_original28']=[uid for uid,q in rows.items() if q['cohort']=='stream34' and uid.startswith('ovo:')]
    subsets['stream_supported16']=[uid for uid in subsets['stream_original28'] if labels[uid]['verdict']=='supported']
    subsets['stream_manual6']=[uid for uid,q in rows.items() if q['cohort']=='stream34' and not uid.startswith('ovo:')]
    # Condition on the NEW unchanged48 reference; old native outcomes are not an oracle.
    for base in ['natural171','T98','primary269','stream_supported16','stream_manual6']:
        for success in (True,False):
            subsets[base+('_raw48_correct' if success else '_raw48_wrong')]=[
                uid for uid in subsets[base] if lookup[uid,'full48']['correct']==success]
    tables={}
    for name,ids in subsets.items():
        scores={a:dict(correct=sum(lookup[uid,a]['correct'] for uid in ids),denominator=len(ids),
            missing=sum(lookup[uid,a]['status']=='missing' for uid in ids),
            format_errors=sum(lookup[uid,a]['status']=='format_error' for uid in ids)) for a in ARMS}
        paired={}
        for candidate,reference in [('odd24','full48'),('tome24','full48'),('tome24','odd24')]:
            paired[candidate+'_vs_'+reference]=dict(
                corrections=[uid for uid in ids if lookup[uid,candidate]['correct'] and not lookup[uid,reference]['correct']],
                new_losses=[uid for uid in ids if lookup[uid,reference]['correct'] and not lookup[uid,candidate]['correct']])
        tables[name]=dict(scores=scores,paired=paired)
    references={r['id']:r for r in read(OUT/'raw48_references_LOCAL_ONLY.json')}
    replay=dict(denominator=269,matched=sum(lookup[uid,'full48']['generated_ids']==ref['generated_ids'] for uid,ref in references.items()),
        mismatch_ids=[uid for uid,ref in references.items() if lookup[uid,'full48']['generated_ids']!=ref['generated_ids']],
        notes='check unchanged baseline drift separately; pairwise reference, not native upper bound')
    save(destination,dict(status='strict_fixed_analysis',tables=tables,details=details,old_raw48_ID_replay=replay,
        expected_answers=909,actual_answers=len(predictions),
        label_sha256=sha(OUT/'labels_LOCAL_ONLY.json'),protocol_sha256=sha(OUT/'inputs/protocol.json'),
        limits=['opened development only; not heldout','fixed odd is a control, not semantic importance selection',
            'same24 source coordinates separate selection from changed values; not a new architecture claim',
            'raw48 vs24 changes input length and language-axis positions; no equal-budget claim',
            'finite full-prefix archive is diagnostic cost, not bounded deployed memory',
            'single shared process timing descriptive, no stable end-to-end speed claim']))
    print(dict(answers=len(predictions),expected=909,subsets=len(subsets)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.results,a.out)
