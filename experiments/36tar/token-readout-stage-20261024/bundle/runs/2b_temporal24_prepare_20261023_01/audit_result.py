"""Independent result integrity/count/baseline and media-cluster delta audit."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,random
OUT=Path(__file__).parent
RESULT=OUT/'results-218029'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={
'answers.jsonl':'c59450c7a8ed35c98da13b88815d1bb3b573019e81c6b76daff2452daa0cc303',
'events.jsonl':'70931e2ef122847fad4441254711ac743f72475ec0e88bee71874e1c347fda19',
'tokens.jsonl':'71d14d2a77963ed6234e354d80d7807bbadec8a78e445467b2ec8faf42c9ccdd',
'loaded.json':'51bcb0b297dbe6501f164a3bbad460b2bbd61f44e199ddf68bd67d538b40d6b5',
'completed.json':'fd7685d8a851270434a7c266762b90fd8547b25a66ba439fb5f952c84999f96b'}
assert all(sha(RESULT/n)==v for n,v in expected.items())
done=read(RESULT/'completed.json'); answers=lines(RESULT/'answers.jsonl')
rows={q['id']:q for q in read(OUT/'inputs/questions.json')}
groups={g['id']:g for g in read(OUT/'inputs/groups.json')}
arms=['odd24','change24','mix24']
lookup={(p['id'],p['arm']):p for p in answers}
assert len(lookup)==len(answers)==done['answers']==909
assert set(lookup)=={(uid,a) for uid in rows for a in arms}
assert done['vision_encodings']==3981 and done['pretrained_loads']==1
assert done['execution_lock_sha256']==sha(OUT/'inputs/execution_lock.json')
tokens=defaultdict(list)
for t in lines(RESULT/'tokens.jsonl'):tokens[t['query_id'],t['arm']].append(t)
assert sum(map(len,tokens.values()))==done['generated_tokens']==1922
for key,p in lookup.items():
    assert p['status']=='ok' and p['main_unchanged'] and p['archive_unchanged']
    assert p['visual_tokens']==rows[key[0]]['pairs']*(48 if key[1]=='full48' else 24)
    ts=tokens[key]
    assert [t['token_index'] for t in ts]==list(range(len(ts)))
    assert [t['token_id'] for t in ts]==p['generated_ids'] and len(ts)<=8
events=lines(RESULT/'events.jsonl')
ends=[e for e in events if e.get('kind')=='group_end']
assert len(ends)==112 and {e['group'] for e in ends}==set(groups)
for e in ends:
    pairs=e['archive']['pairs']
    assert pairs<=320
    assert e['archive']['selector_bytes']==393216
    assert all(e['archive']['tokens'][a]==pairs*(48 if a=='full48' else 24) for a in arms)
refs=read(OUT/'historical_references_LOCAL_ONLY.json')['records']
assert len(refs)==303 and all(lookup[r['id'],r['arm']]['generated_ids']==r['generated_ids'] for r in refs)
score=read(OUT/'answer_results.json')
detail={(d['id'],d['arm']):d for d in score['details']}
media=defaultdict(list)
for uid,q in rows.items():
    if q['cohort']!='stream34':media[groups[q['group']]['sha256']].append(uid)
rng=random.Random(20261023); intervals={}
for control in ('odd24','mix24'):
    clusters=[(sum(int(detail[u,'change24']['answer_correct'])-int(detail[u,control]['answer_correct']) for u in us),len(us)) for us in media.values()]
    values=[]
    for _ in range(5000):
        sample=rng.choices(clusters,k=len(clusters))
        values.append(sum(d for d,n in sample)/sum(n for d,n in sample))
    values.sort()
    intervals[control]=dict(delta_pp=100*sum(d for d,n in clusters)/sum(n for d,n in clusters),
        percentile95_pp=[100*values[125],100*values[4874]],media_SHA_clusters=len(clusters),
        note='media groups, not verified independent filming sessions; opened development only')
receipt=dict(status='passed',raw_SHA256=expected,answers=909,token_callbacks=1922,
    source_protocol_groups=112,old_generated_ID_matches=303,weight_loads=1,
    archive_and_main_isolation_recorded_all=True,visual_token_counts_all_match=True,
    primary269_change24_cluster_delta=intervals,answer_results_SHA256=sha(OUT/'answer_results.json'))
with (OUT/'result_audit.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
