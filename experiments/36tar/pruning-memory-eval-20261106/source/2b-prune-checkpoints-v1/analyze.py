"""Paired development evidence; retain every checkpoint, never rebrand as held-out testing."""
import argparse,collections,json,statistics
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];P=R/'runs/prune_checkpoints_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def summarize(rows,questions,arms):
    index={(r['id'],r['arm']):r for r in rows};qs={q['id']:q for q in questions}
    assert len(index)==len(rows)==len(qs)*len(arms)
    assert set(index)=={(qid,arm) for qid in qs for arm in arms}
    videos=sorted({q['video_id'] for q in questions});summary={}
    for r in rows:
        assert r['shift']==0 and r['role']=='development' and r['video_id']==qs[r['id']]['video_id']
        assert r['correct']==(r['original_selected_index']==qs[r['id']]['answer_index'])
        assert np.isfinite(r['ce'])
    for arm in arms:
        rr=[index[q,arm] for q in qs]
        summary[arm]=dict(n=len(rr),correct=sum(r['correct'] for r in rr),ce_mean=statistics.mean(r['ce'] for r in rr),
            format_only_letter=sum(r['format_only_letter'] for r in rr),option_argmax_correct=sum(r['option_argmax_correct'] for r in rr),
            by_type={t:dict(n=sum(q['type']==t for q in questions),correct=sum(index[q['id'],arm]['correct'] for q in questions if q['type']==t)) for t in sorted({q['type'] for q in questions})})
    paired={};counts=np.array([sum(q['video_id']==v for q in questions) for v in videos])
    draw=np.random.default_rng(20261102).integers(0,len(videos),size=(2000,len(videos)))
    for arm in arms:
        if arm=='step1237':continue
        gain=[q for q in qs if index[q,arm]['correct'] and not index[q,'step1237']['correct']]
        loss=[q for q in qs if not index[q,arm]['correct'] and index[q,'step1237']['correct']]
        d=np.array([sum(int(index[q['id'],arm]['correct'])-int(index[q['id'],'step1237']['correct']) for q in questions if q['video_id']==v) for v in videos])
        paired[arm]=dict(reference='step1237',gained_ids=gain,lost_ids=loss,net=len(gain)-len(loss),
            source_bootstrap_95_pp=np.quantile(100*d[draw].sum(1)/counts[draw].sum(1),[.025,.975]).tolist())
    ranked=sorted(arms,key=lambda a:(-summary[a]['correct'],summary[a]['ce_mean'],int(a[4:])))
    return dict(methods=summary,paired_to_final=paired,development_ranking=ranked,
        limitation='Exploratory same-development comparison. No multiplicity-adjusted superiority, unseen generalization, automatic promotion or independent timing.')
def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    assert read(a.run/'completed.json')['status']=='complete' and read(a.run/'endpoint_gate.json')==dict(passed=True,replayed=690)
    rows=lines(a.run/'answers.jsonl');spec=read(P/'protocol.json');questions=read(P/'questions.json')
    result=summarize(rows,questions,spec['arms'])
    refs={(r['id'],'step0' if r['arm']=='mask_initial' else 'step1237'):r for r in read(P/'reference.json')}
    for r in rows:
        if r['arm'] in ('step0','step1237'):
            assert r['endpoint_replay'] is True
            assert all(r[k]==refs[r['id'],r['arm']][k] for k in ('generated_ids','input_ids_sha256','input_tokens','selected_index','correct'))
    selects=lines(a.run/'selections.jsonl');index={(r['video_id'],r['arm']):r for r in selects}
    assert len(index)==len(selects)==200
    changes={}
    for arm in spec['arms']:
        values=[]
        for vid in sorted({q['video_id'] for q in questions}):
            aa=index[vid,arm]['pairs'];bb=index[vid,'step0']['pairs'];assert len(aa)==len(bb)
            for x,y in zip(aa,bb):
                assert x['pair']==y['pair'] and x['begin']==y['begin'] and x['end']==y['end']
                assert len(set(x['ids']))==len(x['ids'])==54 and all(0<=i<108 for i in x['ids'])
                values.append(54-len(set(x['ids'])&set(y['ids'])))
        changes[arm]=dict(pairs=len(values),mean_replaced_of54=statistics.mean(values),different_pairs=sum(v>0 for v in values))
    result['selection_changes_from_initial']=changes;result['run']=read(a.run/'completed.json')
    with (a.run/'analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['methods']))
if __name__=='__main__':main()
