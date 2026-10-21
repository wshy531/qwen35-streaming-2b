"""Prewritten five-arm strict analysis. No denominator or parser relaxation."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
OLD=ROOT/'runs/2b_long_stream_prepare_20261020_01'
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from score_memory_eval import parse_choice
ARMS=('M0_384','M2_384','M0_768','M2_768','S24_384')
PAIRS=(('M0_768','M0_384'),('M2_768','M2_384'),('M2_384','M0_384'),
       ('M2_768','M0_768'),('S24_384','M2_384'),('S24_384','M0_384'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(actual,rows,labels):
    details=[]
    for i,r in rows.items():
        for arm in ARMS:
            a=actual.get((i,arm));text=None if a is None else a['text']
            choice=parse_choice(text,len(r['options']))
            details.append(dict(id=i,arm=arm,source=r['source_path'],text=text,
                predicted_index=choice,truth=labels[i]['answer_index'],correct=choice==labels[i]['answer_index'],
                status='missing' if a is None else 'format_error' if choice is None else 'ok',
                generated_ids=[] if a is None else a['generated_ids']))
    lookup={(d['id'],d['arm']):d for d in details}
    groups=dict(original_all28=[i for i,r in rows.items() if r['cohort']=='official_OVO_unchanged'],
        original_supported16=[i for i,r in rows.items() if r['cohort']=='official_OVO_unchanged' and labels[i]['verdict']=='supported'],
        manual_development6=[i for i,r in rows.items() if r['cohort']=='manual_development'])
    assert list(map(len,groups.values()))==[28,16,6]
    summary={}
    for group,ids in groups.items():
        counts={a:dict(correct=sum(lookup[i,a]['correct'] for i in ids),denominator=len(ids),
            missing=sum(lookup[i,a]['status']=='missing' for i in ids),
            format_errors=sum(lookup[i,a]['status']=='format_error' for i in ids)) for a in ARMS}
        paired={a+'_vs_'+b:dict(corrections=[i for i in ids if lookup[i,a]['correct'] and not lookup[i,b]['correct']],
            losses=[i for i in ids if lookup[i,b]['correct'] and not lookup[i,a]['correct']]) for a,b in PAIRS}
        summary[group]=dict(scores=counts,paired=paired)
    return dict(summary=summary,details=details)


def main(folder,selfcheck=False):
    rows={r['id']:r for r in read(OLD/'stream_requests34.json')}
    labels={r['id']:r for r in read(OLD/'stream_labels34_LOCAL_ONLY.json')}
    if selfcheck:
        actual={(i,a):dict(text=chr(65+labels[i]['answer_index']),generated_ids=[]) for i in rows for a in ARMS}
        r=score(actual,rows,labels)
        assert all(v['correct']==v['denominator'] for g in r['summary'].values() for v in g['scores'].values())
        actual={k:v for k,v in actual.items() if k[1]!='S24_384'};r=score(actual,rows,labels)
        assert all(g['scores']['S24_384']['missing']==g['scores']['S24_384']['denominator'] for g in r['summary'].values())
        print(dict(scoring_contract='passed',synthetic_answers=170,missing_arm_denominators_preserved=True));return
    snaps={s['query_id']:s for s in read(OLD/'stream_query_snapshots34.json')};actual={};previous={}
    for line in (OLD/'gpu_217776/memory/events.jsonl').open(encoding='utf-8'):
        e=json.loads(line)
        if e['event']['kind']=='query' and e['arm'] in ('M0','M2'):
            previous[e['event']['query_id'],e['arm']+'_384']=e['answer']['generated_ids']
    for line in (folder/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line)
        if e['event']['kind']!='query':continue
        i=e['event']['query_id'];assert i in rows
        assert e['committed_frames_sha256']==snaps[i]['committed_frames_sha256']
        assert e['event']['committed_through_s']<=rows[i]['query_time_s']
        assert e['answer']['question_did_not_change_main_or_banks']
        assert set(e['answer']['arms'])==set(ARMS)
        for a,v in e['answer']['arms'].items():assert (i,a) not in actual;actual[i,a]=v
    result=score(actual,rows,labels)
    result.update(actual_records=len(actual),expected_records=170,
        old_baseline_generated_ID_replay={a:dict(matched=sum(actual.get((i,a),{}).get('generated_ids')==previous[i,a] for i in rows),
            denominator=34,mismatch_ids=[i for i in rows if actual.get((i,a),{}).get('generated_ids')!=previous[i,a]]) for a in ('M0_384','M2_384')},
        source_lock={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),OLD/'stream_labels34_LOCAL_ONLY.json',ROOT/'experiments/2b-natural-eval-v1/score_memory_eval.py')},
        limits=['six reused development sources; no formal heldout capability claim','capacity control has double payload',
            'format errors kept; no relaxed scoring','shared process not per-arm speed','baseline drift must be explained'])
    with (OUT/'strict_results.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path);p.add_argument('--selfcheck',action='store_true')
    a=p.parse_args();assert a.selfcheck or a.results;main(a.results,a.selfcheck)
