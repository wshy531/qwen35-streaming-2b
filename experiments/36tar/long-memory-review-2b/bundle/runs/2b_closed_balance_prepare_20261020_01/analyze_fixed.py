"""Strict three-arm scores with all original denominators and old replay checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
OLD=ROOT/'runs/2b_long_stream_prepare_20261020_01'
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from score_memory_eval import parse_choice


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main(folder):
    rows={q['id']:q for q in read(OLD/'stream_requests34.json')}
    labels={q['id']:q for q in read(OLD/'stream_labels34_LOCAL_ONLY.json')}
    snapshots={s['query_id']:s for s in read(OLD/'stream_query_snapshots34.json')}
    previous={};actual={}
    for line in (OLD/'gpu_217776/memory/events.jsonl').open(encoding='utf-8'):
        e=json.loads(line)
        if e['event']['kind']=='query' and e['arm'] in ('M0','M2'):
            previous[e['event']['query_id'],e['arm']]=e['answer']['generated_ids']
    for line in (folder/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line)
        if e['event']['kind']!='query':continue
        ident=e['event']['query_id'];assert ident in rows
        assert e['committed_frames_sha256']==snapshots[ident]['committed_frames_sha256']
        assert e['event']['committed_through_s']<=rows[ident]['query_time_s']
        assert e['answer']['question_did_not_change_main_or_banks']
        for arm,answer in e['answer']['arms'].items():
            assert arm in ('M0','M2','A06') and (ident,arm) not in actual
            actual[ident,arm]=answer
    details=[]
    for ident,row in rows.items():
        for arm in ('M0','M2','A06'):
            a=actual.get((ident,arm));text=None if a is None else a['text']
            choice=parse_choice(text,len(row['options']))
            details.append(dict(id=ident,arm=arm,source=row['source_path'],text=text,
                predicted_index=choice,truth=labels[ident]['answer_index'],correct=choice==labels[ident]['answer_index'],
                status='missing' if a is None else 'format_error' if choice is None else 'ok',
                generated_ids=[] if a is None else a['generated_ids']))
    lookup={(d['id'],d['arm']):d for d in details};groups={
        'original_all28':[i for i,r in rows.items() if r['cohort']=='official_OVO_unchanged'],
        'original_supported16':[i for i,r in rows.items() if r['cohort']=='official_OVO_unchanged' and labels[i]['verdict']=='supported'],
        'manual_development6':[i for i,r in rows.items() if r['cohort']=='manual_development']}
    summary={}
    for name,ids in groups.items():
        scores={a:dict(correct=sum(lookup[i,a]['correct'] for i in ids),denominator=len(ids),
            missing=sum(lookup[i,a]['status']=='missing' for i in ids),
            format_errors=sum(lookup[i,a]['status']=='format_error' for i in ids)) for a in ('M0','M2','A06')}
        pairs={}
        for base in ('M0','M2'):
            pairs['A06_vs_'+base]=dict(corrections=[i for i in ids if lookup[i,'A06']['correct'] and not lookup[i,base]['correct']],
                losses=[i for i in ids if lookup[i,base]['correct'] and not lookup[i,'A06']['correct']])
        summary[name]=dict(scores=scores,paired=pairs)
    replay={a:dict(matched=sum(actual.get((i,a),{}).get('generated_ids')==previous[i,a] for i in rows),
        denominator=34,mismatch_ids=[i for i in rows if actual.get((i,a),{}).get('generated_ids')!=previous[i,a]]) for a in ('M0','M2')}
    result=dict(summary=summary,details=details,old_baseline_generated_ID_replay=replay,
        actual_records=len(actual),expected_records=102,
        limits=['six already-open sources, not formal20/160','all28 includes ambiguous/not-established questions',
          'no relaxed rescoring, no speed claim from shared process','depth reduction alone is not ability',
          'baseline replay differences must be explained before attribution'])
    with (OUT/'strict_results.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(summary=summary,baseline_replay=replay),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);main(p.parse_args().results)
