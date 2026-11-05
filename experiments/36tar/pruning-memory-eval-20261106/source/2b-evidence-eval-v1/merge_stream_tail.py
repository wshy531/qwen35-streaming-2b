"""Combine disjoint cells; retain the original timed-out run unchanged."""
import argparse,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def key(r):return (r['id'],r['arm'],r['protocol'],r['shift'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--tail',type=Path,required=True);a=p.parse_args()
    a.tail=a.tail.resolve()
    original=R/'runs/evidence-stream-220068';tail=read(a.tail/'completed.json');assert tail['status']=='complete_missing_tail_only'
    old=lines(original/'answers.jsonl');new=lines(a.tail/'answers.jsonl');assert not {key(r) for r in old}&{key(r) for r in new}
    qs={q['id']:q for q in read(P/'panel_media_bound.json') if q['dataset']=='StreamingBench'}
    rows=old+new;expected={(q,arm,proto,shift) for q in qs for arm in ('full','fixed12','heuristic11') for proto in ('A','B') for shift in (0,2)}
    assert len(rows)==len({key(r) for r in rows})==360 and {key(r) for r in rows}==expected
    for r in rows:
        q=qs[r['id']];assert r['source_id']==q['source_id'] and r['query_time_s']==q['query_time_s']
        assert len(r['visible_pairs'])<=8 and r['source_state']['last_committed_pts']<=q['query_time_s']+1e-6
        r['execution_part']='original_serial' if key(r) in {key(x) for x in old} else 'missing_tail_V0_auto4'
    out=R/'runs/evidence-stream-merged-20261102-01'
    merged=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=key))
    if out.exists():
        assert set(p.name for p in out.iterdir())=={'answers.jsonl'}
        assert (out/'answers.jsonl').read_text(encoding='utf-8')==merged
    else:
        out.mkdir();(out/'answers.jsonl').write_text(merged,encoding='utf-8')
    result=dict(status='complete_merged_two_execution_parts',answers=360,original_answers=len(old),tail_answers=len(new),
        original_run=str(original.relative_to(R)),tail_run=str(a.tail.relative_to(R)),model_loads=2,
        generated_tokens=sum(len(r['generated_ids']) for r in rows),
        vision_calls=read(original/'failed.json')['vision_calls']+tail['vision_calls'],
        original_timeout_preserved=True,API=0,training=0,exact_cells_and_causal_cutoffs_checked=True,
        timing_limit='two decoders and repeated unfinished source encoding; not standalone algorithm cost')
    (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
