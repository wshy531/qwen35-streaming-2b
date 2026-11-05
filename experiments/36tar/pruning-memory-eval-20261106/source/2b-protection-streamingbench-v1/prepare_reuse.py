"""Reuse audited causal source states; never reuse answers across changed prompts."""
from common import *
raw=R/'runs/evidence-scale-stream-tail-220170/answers.jsonl'
rows=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()]
qs=read(P/'questions.json');expected={}
for q in qs:
    old=[r for r in rows if r['id']==q['id']]
    assert len(old)==5 and all(r['source_state']==old[0]['source_state'] for r in old)
    assert old[0]['source_state']['last_committed_pts']<=q['query_time_s']+1e-6
    expected[q['id']]=dict(state=old[0]['source_state'])
totals={}
for q in qs:totals[q['source_id']]=max(totals.get(q['source_id'],0),expected[q['id']]['state']['complete_pairs'])
save(P/'causal_reference.json',dict(source_file=str(raw.relative_to(R)),source_sha256=sha(raw),prefixes=expected,
    pixel_check_ids=[qs[0]['id'],qs[1]['id']],vision_pairs=sum(totals.values()),
    caveat='Reuse original V0 causal source states only. Answers have a different prompt/token limit and are not new answer baselines. Full video SHA required for all files; exact fresh pixel checks for two predeclared sources; GPU logs all actual pixel digests.'))
print('causal references',len(expected),'visual pairs',sum(totals.values()))
