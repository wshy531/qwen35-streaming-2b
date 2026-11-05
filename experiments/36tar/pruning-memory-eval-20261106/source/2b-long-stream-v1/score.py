"""Predefined strict, stratified analysis. Missing answers stay in denominator."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from inventory_local import ROOT,OUT
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from score_memory_eval import parse_choice


def analyze(result_dir):
    def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
    rows={q['id']:q for q in read('stream_requests34.json')}
    labels={q['id']:q for q in read('stream_labels34_LOCAL_ONLY.json')}
    arms=['native','main_only','M0','M2','text_only'];pred={};details=[]
    expected={s['query_id']:s for s in read('stream_query_snapshots34.json')}
    for stage in ('native','memory'):
        path=result_dir/stage/'events.jsonl'
        if not path.exists():continue
        for line in path.read_text(encoding='utf-8').splitlines():
            item=json.loads(line)
            if item['event']['kind']!='query':continue
            ident=item['event']['query_id'];arm=item['arm'];assert ident in rows and arm in arms
            assert (ident,arm) not in pred
            assert item['committed_frames_sha256']==expected[ident]['committed_frames_sha256']
            assert item['answer']['observation_through_s']<=rows[ident]['query_time_s']
            pred[ident,arm]=item
    for ident,row in rows.items():
        for arm in arms:
            p=pred.get((ident,arm));text=None if p is None else p['answer']['text']
            choice=parse_choice(text,len(row['options']))
            details.append(dict(id=ident,arm=arm,source_path=row['source_path'],cohort=row['cohort'],
                verdict=labels[ident]['verdict'],status='missing' if p is None else 'format' if choice is None else 'answered',
                text=text,prediction=choice,truth=labels[ident]['answer_index'],correct=choice==labels[ident]['answer_index'],
                available_committed_through_s=expected[ident]['committed_through_s'],
                text_control_does_not_see_video=arm=='text_only'))
    cohorts=dict(official_all28=[i for i,r in rows.items() if r['cohort']=='official_OVO_unchanged'],
        official_supported16=[i for i,r in rows.items() if r['cohort']=='official_OVO_unchanged' and labels[i]['verdict']=='supported'],
        manual_development6=[i for i,r in rows.items() if r['cohort']=='manual_development'])
    lookup={(d['id'],d['arm']):d for d in details};summary={}
    for name,ids in cohorts.items():
        scores={a:dict(correct=sum(lookup[i,a]['correct'] for i in ids),total=len(ids),
            missing=sum(lookup[i,a]['status']=='missing' for i in ids),
            format_errors=sum(lookup[i,a]['status']=='format' for i in ids)) for a in arms}
        pairs={}
        for a,b in [('M2','M0'),('M2','native'),('M0','native'),('M2','text_only'),('M2','main_only')]:
            pairs[a+'_vs_'+b]=dict(corrections=[i for i in ids if lookup[i,a]['correct'] and not lookup[i,b]['correct']],
                losses=[i for i in ids if lookup[i,b]['correct'] and not lookup[i,a]['correct']])
        bysource={}
        for source in sorted({rows[i]['source_path'] for i in ids}):
            subset=[i for i in ids if rows[i]['source_path']==source]
            bysource[source]={a:dict(correct=sum(lookup[i,a]['correct'] for i in subset),total=len(subset)) for a in arms}
        summary[name]=dict(scores=scores,paired=pairs,per_source=bysource)
    return dict(summary=summary,details=details,actual_records=len(pred),expected_records=170,
        limitations=['development/integration only; six reused sources and single reviewer',
          'all28 includes12 ambiguous/not-established originals, not a clean accuracy claim',
          'manual6 separate from original questions; delayed evidence is not a necessity proof',
          'causal replay is not wall-clock playback; no speed or stable generalization conclusion'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=analyze(a.results)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
