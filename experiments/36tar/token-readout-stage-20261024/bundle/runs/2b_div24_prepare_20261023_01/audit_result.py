"""Independent result integrity/count/baseline and media-cluster delta audit."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,random
OUT=Path(__file__).parent
RESULT=OUT/'results-218021'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={
'answers.jsonl':'07c06c77b7ca91fdbcf8744e77ddc1a5bdad48571c382bad14014df267d63be0',
'events.jsonl':'29ab22669a5e4658afb7588dcf972358952480f9600f3d0cf14775e07e41d32b',
'tokens.jsonl':'e74443e58cdc815453aef47f94dab2f1152a790ca73c7a1991d57260c44390f6',
'loaded.json':'d6c4a869dafaea5bbe1cedaed0f78c07803b6e95ba4e20798884c550fd2b0548',
'completed.json':'45f4878aa9238095aaa27cf83752f0900f54e891744be7cca4cb5414031b48fa'}
assert all(sha(RESULT/n)==v for n,v in expected.items())
done=read(RESULT/'completed.json'); answers=lines(RESULT/'answers.jsonl')
rows={q['id']:q for q in read(OUT/'inputs/questions.json')}
groups={g['id']:g for g in read(OUT/'inputs/groups.json')}
arms=['full48','odd24','grid24','tome24','div24']
lookup={(p['id'],p['arm']):p for p in answers}
assert len(lookup)==len(answers)==done['answers']==1515
assert set(lookup)=={(uid,a) for uid in rows for a in arms}
assert done['vision_encodings']==3981 and done['pretrained_loads']==1
assert done['execution_lock_sha256']==sha(OUT/'inputs/execution_lock.json')
tokens=defaultdict(list)
for t in lines(RESULT/'tokens.jsonl'):tokens[t['query_id'],t['arm']].append(t)
assert sum(map(len,tokens.values()))==done['generated_tokens']==3155
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
    assert all(e['archive']['tokens'][a]==pairs*(48 if a=='full48' else 24) for a in arms)
refs=read(OUT/'historical_references_LOCAL_ONLY.json')['records']
assert len(refs)==909 and all(lookup[r['id'],r['arm']]['generated_ids']==r['generated_ids'] for r in refs)
score=read(OUT/'answer_results.json')
detail={(d['id'],d['arm']):d for d in score['details']}
media=defaultdict(list)
for uid,q in rows.items():
    if q['cohort']!='stream34':media[groups[q['group']]['sha256']].append(uid)
rng=random.Random(20261023); intervals={}
for control in ('full48','odd24','grid24','tome24'):
    clusters=[(sum(int(detail[u,'div24']['answer_correct'])-int(detail[u,control]['answer_correct']) for u in us),len(us)) for us in media.values()]
    values=[]
    for _ in range(5000):
        sample=rng.choices(clusters,k=len(clusters))
        values.append(sum(d for d,n in sample)/sum(n for d,n in sample))
    values.sort()
    intervals[control]=dict(delta_pp=100*sum(d for d,n in clusters)/sum(n for d,n in clusters),
        percentile95_pp=[100*values[125],100*values[4874]],media_SHA_clusters=len(clusters),
        note='media groups, not verified independent filming sessions; opened development only')
receipt=dict(status='passed',raw_SHA256=expected,answers=1515,token_callbacks=3155,
    source_protocol_groups=112,old_generated_ID_matches=909,weight_loads=1,
    archive_and_main_isolation_recorded_all=True,visual_token_counts_all_match=True,
    primary269_div24_cluster_delta=intervals,answer_results_SHA256=sha(OUT/'answer_results.json'))
with (OUT/'result_audit.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
