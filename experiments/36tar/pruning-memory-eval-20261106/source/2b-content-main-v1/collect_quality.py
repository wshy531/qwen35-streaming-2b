"""Source-atomic collection: retain all raw records; incomplete sources never count.

This can collect successful shards, or assess a bounded timeout without pretending
the job completed. A later explicitly launched completion must restart an entire
source, preserving its causal state. No question/result-based source selection.
"""
import argparse,json
from pathlib import Path
from collections import Counter
from common import ROOT,PREP,read,save,sha,MODES
def collect(folders,out):
    qs=read(PREP/'requests.json');byvid={v:{q['question_id'] for q in qs if q['video_id']==v} for v in {q['video_id'] for q in qs}}
    expected=read(PREP/'expected_prefixes.json');refs=read(PREP/'references.json');allrows=[];receipts=[];seen=set()
    for folder in folders:
        for path in Path(folder).glob('quality_*/answers.jsonl'):
            raw=[];invalid_tail=False
            for line in path.read_text(encoding='utf-8').splitlines():
                try:raw.append(json.loads(line))
                except json.JSONDecodeError:invalid_tail=True;break
            grouped={v:[r for r in raw if r['video_id']==v] for v in {r['video_id'] for r in raw}}
            accepted=[];incomplete=[]
            for vid,rows in grouped.items():
                keys={(r['question_id'],r['arm']) for r in rows}
                wanted={(q,m) for q in byvid[vid] for m in MODES}
                if keys!=wanted or len(keys)!=len(rows):incomplete.append(vid);continue
                if vid in seen:raise ValueError('duplicate complete source; manual provenance resolution required')
                for r in rows:
                    for key in read(PREP/'protocol.json')['original_prefix_identity_keys']:
                        assert r['query_prefix'][key]==expected[r['question_id']][key]
                    ref=refs.get(r['question_id']+'|'+r['arm'])
                    if ref:
                        for key in ref:assert r[key]==ref[key]
                    assert r['branch_released'] and r['read_state_unchanged']
                seen.add(vid);accepted.append(vid);allrows+=rows
            receipts.append(dict(path=str(path),sha256=sha(path),accepted_sources=accepted,
                incomplete_sources=incomplete,invalid_trailing_JSON_retained_in_raw=invalid_tail))
    missing=sorted(set(byvid)-seen)
    result=dict(status='complete' if not missing else 'partial',sources=len(seen),answers=len(allrows),
        remaining_sources=missing,receipts=receipts,no_score_based_filtering=True,
        all_raw_failures_retained=True,source_atomic=True)
    out.mkdir(exist_ok=False)
    save(out/'collection.json',result)
    with (out/'answers.jsonl').open('x',encoding='utf-8') as f:
        for r in allrows:f.write(json.dumps(r)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='receipts'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folders',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();collect(a.folders,a.out)
