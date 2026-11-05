"""Freeze missing keys without looking at labels or answer correctness."""
from fair_common import *
import collections

T=R/'runs/protection_stream_tail_prepare_20261103_01'
T.mkdir(exist_ok=True)
old=R/'runs/protect-stream-220270'
rows=[json.loads(x) for x in (old/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
qs=read(P/'questions.json');refs=read(P/'media_verified.json')['prefixes']
index={(r['id'],r['arm']):r for r in rows}
assert len(index)==len(rows)==532
complete={q['id'] for q in qs if all((q['id'],a) in index for a in ARMS)}
assert len(complete)==133 and len(complete)*4==len(rows)
missing=[q for q in qs if q['id'] not in complete]
sources={q['source_id'] for q in missing}
# Choose by prefix duration, not results; require history beyond the recent window.
control=min((q for q in qs if q['id'] in complete and q['source_id'] not in sources
             and refs[q['id']]['state']['complete_pairs']>=12),key=lambda q:(q['query_time_s'],q['id']))
selected=[control]+missing
groups=collections.defaultdict(list)
for q in selected:groups[q['source_id']].append(q)
pairs=sum(max(refs[q['id']]['state']['complete_pairs'] for q in group) for group in groups.values())
ordered=sorted(qs,key=lambda q:(q['source_id'],q['query_time_s'],q['id']))
manifest=dict(parent='protect-stream-220270',parent_answers_sha256=sha(old/'answers.jsonl'),
    missing_ids=[q['id'] for q in missing],control_id=control['id'],
    control_source=control['source_id'],questions=selected,
    global_indices={q['id']:i for i,q in enumerate(ordered)},
    vision_pairs=pairs,formal_answers=68,replayed_answers=4,warmups=4,
    max_answers=76,max_generated_tokens=1216,wall_minutes=25,internal_minutes=23,
    expected_control={a:index[control['id'],a] for a in ARMS})
assert len(missing)==17 and len(sources)==16 and len(groups)==17
save(T/'manifest.json',manifest)

# Preserve old worker; mechanically specialize only scheduling and completion checks.
src=(Path(__file__).parent/'worker_v3.py').read_text(encoding='utf-8')
def change(a,b):
    global src
    assert src.count(a)==1,(a,src.count(a))
    src=src.replace(a,b)
change('from fair_common import *','from tail_common import *')
change("verify();auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['lock_sha256']==sha(P/'execution_lock4.json')",
       "verify_tail();auth=read(T/'GPU_authorization.json');assert auth['approved'] and auth['lock_sha256']==sha(T/'execution_lock.json'); manifest=read(T/'manifest.json')")
change('88*60','23*60');change('88-minute stop line','23-minute stop line')
change('64*2**20','16*2**20')
change("for q in read(P/'questions.json'):groups[q['source_id']].append(q)","for q in manifest['questions']:groups[q['source_id']].append(q)")
change('enumerate(sorted(groups.items()))',"enumerate(sorted(groups.items(),key=lambda item:(item[0]!=manifest['control_source'],item[0])))")
change("off=counts['answers']//4%4","off=manifest['global_indices'][q['id']]%4")
start=src.index('                        if index<8 and ')
end=src.index('                finally:driver.close();session.close()',start)
src=src[:start]+'''                        if q['id']==manifest['control_id']:
                            for arm in ARMS:
                                previous=manifest['expected_control'][arm];current=responses[arm]
                                for key in ('generated_ids','bounded_context_ids_sha256','position_sha256'):
                                    assert current.get(key)==previous.get(key),('endpoint',arm,key)
                                assert state==previous['source_state'] and d.hexdigest()==previous['pixel_sha256']
                            assert len(h)>0
                            counts['numerical']=4
                            save(out/'endpoint_replay.json',dict(passed=True,answers=4,id=q['id'],scope='generated tokens, input IDs, positions, pixels, source state; not logits'))
'''+src[end:]
change("counts['answers']==600 and counts['vision']==expected['vision_pairs']","counts['answers']==72 and counts['vision']==manifest['vision_pairs']")
change("counts['generated_tokens']<=9792 and counts['numerical']==8","counts['generated_tokens']<=1216 and counts['numerical']==4")
change("verify();save(out/'completed.json',dict(status='complete',**counts,seconds=time.monotonic()-start,model_loads=1,training=0))",
       "verify_tail();save(out/'completed.json',dict(status='complete',scope='tail_only',parent='protect-stream-220270',new_answers=68,replayed_answers=4,**counts,seconds=time.monotonic()-start,model_loads=1,training=0))")
target=Path(__file__).parent/'worker_tail.py'
target.write_text(src,encoding='utf-8',newline='\n')
print(json.dumps(dict(missing=17,missing_sources=16,control_time=control['query_time_s'],vision_pairs=pairs,answers=76)))
