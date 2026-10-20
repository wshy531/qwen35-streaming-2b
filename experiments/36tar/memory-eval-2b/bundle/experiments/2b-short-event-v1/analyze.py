"""Local-only, fixed scoring and paired source analysis; labels never reach workers."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scoring import measure
from prepare import ARMS, DIAG_ARMS

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def lines(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()]

def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2)
        f.write('\n')

def summarize(items):
    groups = sorted({r['source_group_id'] for r in items})
    result = dict(questions=len(items), source_groups=len(groups),
                  strict_correct=sum(r['strict_correct'] for r in items),
                  content_correct=sum(r['content_correct'] for r in items),
                  format_compliant=sum(r['format_compliant'] for r in items),
                  unparsable=sum(not r['content_parseable'] for r in items))
    for metric in ['strict_correct','content_correct']:
        result[metric+'_macro'] = float(np.mean([
            np.mean([r[metric] for r in items if r['source_group_id']==g]) for g in groups]))
    return result

def paired(ref, candidate, metric):
    a,b = {r['id']:r for r in ref},{r['id']:r for r in candidate}
    if a.keys()!=b.keys() or not a:
        raise ValueError('complete matching paired IDs required')
    groups=sorted({r['source_group_id'] for r in ref})
    deltas=np.array([np.mean([int(b[k][metric])-int(a[k][metric]) for k in a
                             if a[k]['source_group_id']==g]) for g in groups])
    draws=np.random.default_rng(20261020).integers(0,len(groups),(2000,len(groups)))
    return dict(corrections=sum(not a[k][metric] and b[k][metric] for k in a),
                new_losses=sum(a[k][metric] and not b[k][metric] for k in a),
                macro_delta=float(deltas.mean()),
                interval95=np.percentile(deltas[draws].mean(1),[2.5,97.5]).tolist(),
                interpretation='descriptive opened-development comparison; no multiplicity-adjusted discovery claim')

def analyze(run,results):
    key={r['id']:r for r in read(run/'scorer_rows_LOCAL_ONLY.json')}
    strata={r['id']:r for r in read(run/'strata_local.json')}
    main=lines(results/'main/predictions.jsonl')
    diag=lines(results/'diagnostic/predictions.jsonl')
    if len(main)!=1197 or len(diag)!=45:
        raise ValueError('incomplete run retained; do not report partial run as final')
    for records, arms, n in [(main,ARMS,171),(diag,DIAG_ARMS,9)]:
        if len({(r['id'],r['arm']) for r in records})!=len(records):
            raise ValueError('duplicate predictions')
        if {r['arm'] for r in records}!=set(arms):
            raise ValueError('unexpected arm')
        if any(sum(r['arm']==a for r in records)!=n for a in arms):
            raise ValueError('wrong per-arm denominator')
    details=[]
    for stage,records in [('main',main),('diagnostic',diag)]:
        for p in records:
            k=key[p['id']]
            text=p['text'] if p['status']=='ok' else ''
            details.append(dict(strata[p['id']],stage=stage,arm=p['arm'],text=p['text'],
                                status=p['status'],
                                **measure(text,len(k['options']),k['answer_index'])))
    # No filtering by model correctness: all171 and all9 stay in the denominators.
    tables={}
    for cohort in ['public74','open47','new50']:
        for layer in ['all','FT','C','pressure','old_outside_recent8','recent_span_covered']:
            rs=[r for r in details if r['stage']=='main' and r['cohort']==cohort]
            if layer=='FT':rs=[r for r in rs if r['stratum'] in ['F','T']]
            if layer=='C':rs=[r for r in rs if r['stratum']=='C']
            if layer=='pressure':rs=[r for r in rs if r['storage_pressure']]
            if layer=='old_outside_recent8':rs=[r for r in rs if r['author_evidence_recent8']=='none_by_time_span']
            if layer=='recent_span_covered':rs=[r for r in rs if r['author_evidence_recent8']=='all_by_time_span']
            if not rs:continue
            armrows={a:[r for r in rs if r['arm']==a] for a in ARMS}
            tables[cohort+'/'+layer]={a:summarize(items) for a,items in armrows.items()}
            tables[cohort+'/'+layer]['paired_B02']={ref:{m:paired(armrows[ref],armrows['B02_cap4'],m)
                for m in ['strict_correct','content_correct']} for ref in ['M0','M1','M2','reservoir4']}
    diagtable={a:summarize([r for r in details if r['stage']=='diagnostic' and r['arm']==a]) for a in DIAG_ARMS}
    previous=read(run/'previous_predictions_LOCAL_ONLY.json')
    prev={(p['id'],p['arm']):p for p in previous}
    replay={}
    for a in ['full_pairwise_reference','M0','M1','M2','text_only']:
        ps=[p for p in main if p['arm']==a]
        replay[a]=dict(cases=len(ps),ids_equal=sum(p.get('generated_ids')==prev[(p['id'],a)].get('generated_ids') for p in ps),
                       input_frames_equal=sum(p.get('input_frames_sha256')==prev[(p['id'],a)].get('input_frames_sha256') for p in ps))
    timelines=read(results/'main/timelines.json')
    capacity={a:dict(max_visual_tokens=max(t['states'][a]['tokens'] for t in timelines))
              for a in ['M0','M1','M2','B02_cap4','reservoir4']}
    if any(x['max_visual_tokens']>384 for x in capacity.values()):
        raise ValueError('capacity violation')
    result=dict(tables=tables,diagnostic=diagtable,baseline_replay=replay,capacity=capacity,
                records=len(details),unique_questions=171,
                decide='no automatic promotion; inspect FT correction/loss, recent regressions and independent T',
                causal_limit='cap also changes closed-root consolidation; does not isolate pure numeric compression',
                reservoir_limit='one fixed random seed; not evidence against all uniform methods')
    save(results/'analysis.json',result)
    with (results/'per_question.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print(json.dumps(dict(records=len(details),baseline_replay=replay)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.results)
