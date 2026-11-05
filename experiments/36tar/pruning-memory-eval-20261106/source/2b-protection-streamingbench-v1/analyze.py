"""Choice-content scoring only; no API or evidence-quality claim."""
import argparse,collections,random,itertools
from common import *
PERCEPTION={'Action Recognition','Attribute Recognition','Object Recognition','Spatial Understanding','Text-Rich Understanding'}
def analyze(out):
    done=read(out/'completed.json');assert done['status']=='complete' and done['answers']==450
    rows=[json.loads(x) for x in (out/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    qs={q['id']:q for q in read(P/'questions.json')};labels=read(R/'runs/evidence_scale_prepare_20261102_01/labels_LOCAL_ONLY.json')
    index={(r['id'],r['arm']):r for r in rows};assert len(index)==len(rows)==450 and set(index)=={(q,a) for q in qs for a in ARMS}
    for (key,arm),r in index.items():
        letter=parse_choice(r['text'],len(qs[key]['options']));selected=None if letter is None else ord(letter)-65
        r.update(selected_index=selected,content_correct=selected==labels[key],parseable=letter is not None)
        assert r['read_state_unchanged'] and r['branch_released'] and not r['source_state']['future_RGB_emitted']
    def score(keys):
        return {a:dict(questions=len(keys),correct=sum(index[q,a]['content_correct'] for q in keys),unparsed=sum(not index[q,a]['parseable'] for q in keys)) for a in ARMS}
    comparisons={}
    for a,b in itertools.combinations(ARMS,2):
        gains=[q for q in qs if index[q,a]['content_correct'] and not index[q,b]['content_correct']]
        losses=[q for q in qs if index[q,b]['content_correct'] and not index[q,a]['content_correct']]
        source=collections.defaultdict(list)
        for q in qs:source[qs[q]['source_id']].append(int(index[q,a]['content_correct'])-int(index[q,b]['content_correct']))
        keys=list(source);rng=random.Random(20261103);values=[]
        for _ in range(2000):
            draw=[source[rng.choice(keys)] for _ in keys];values.append(sum(map(sum,draw))/sum(map(len,draw)))
        values.sort();comparisons[f'{a}_minus_{b}']=dict(gained=gains,lost=losses,net=len(gains)-len(losses),source_bootstrap_exploratory_95=[values[49],values[1949]])
    result=dict(run=done,overall=score(list(qs)),by_type={kind:score([q for q in qs if qs[q]['type']==kind]) for kind in sorted({q['type'] for q in qs.values()})},
        perception_type_candidates=score([q for q in qs if qs[q]['type'] in PERCEPTION]),
        other_types=score([q for q in qs if qs[q]['type'] not in PERCEPTION]),
        by_length={kind:score([q for q in qs if qs[q]['length_bin']==kind]) for kind in sorted({q['length_bin'] for q in qs.values()})},
        comparisons=comparisons,
        measured_QA={a:dict(seconds=sum(index[q,a]['QA_wall_with_audit_seconds'] for q in qs),peak_allocated_bytes=max(index[q,a]['peak_allocated_bytes'] for q in qs)) for a in ARMS},
        limits=['Opened 150 questions/144 media files, not an unseen test','Task type does not certify recent-only answerability','Choice-content scores, not new explanation judging','Shared visual replay and diagnostic audits; not independent end-to-end or edge performance','Original parent-source overlap is not fully audited'])
    save(out/'analysis.json',result)
    with (out/'scored_LOCAL_ONLY.jsonl').open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('overall','perception_type_candidates','other_types','by_type')},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);analyze(p.parse_args().out)
