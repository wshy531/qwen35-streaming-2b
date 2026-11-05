"""Training-side cross-rotation ceiling, not learned selector accuracy."""
import argparse,json,random
from collections import defaultdict
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def mean(x):return sum(x)/len(x)
def compare(a,b):
    key=lambda r:(r['id'],r['shift']);aa={key(r):r for r in a};bb={key(r):r for r in b};assert aa.keys()==bb.keys()
    groups=defaultdict(list)
    for k,r in aa.items():groups[r['video_id']].append((r['gold_probability']-bb[k]['gold_probability'],int(r['correct'])-int(bb[k]['correct'])))
    values=[(mean([x[0] for x in rs]),mean([x[1] for x in rs])) for rs in groups.values()]
    rng=random.Random(20261104);sample=[mean([rng.choice(values)[0] for _ in values]) for _ in range(10000)];sample.sort()
    return dict(probability_gain_question_weighted=mean([aa[k]['gold_probability']-bb[k]['gold_probability'] for k in aa]),
        probability_gain_source_weighted=mean([v[0] for v in values]),probability_source_bootstrap95=[sample[250],sample[9750]],
        positive_sources=sum(v[0]>0 for v in values),negative_sources=sum(v[0]<0 for v in values),
        gained=sum(aa[k]['correct'] and not bb[k]['correct'] for k in aa),lost=sum(bb[k]['correct'] and not aa[k]['correct'] for k in aa))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    completion=read(a.run/'completed.json');assert completion['status']=='completed'
    rows=lines(a.run/'answers.jsonl');decisions=lines(a.run/'decisions.jsonl')
    assert len(rows)==11932 and len(decisions)==314
    fit=[r for r in rows if r['phase']=='fit'];held=[r for r in rows if r['phase']=='confirmation']
    assert len(fit)==10048 and len(held)==1884
    byq=defaultdict(list)
    for r in fit:byq[r['id']].append(r)
    for d in decisions:
        rs=byq[d['id']];assert len({(r['subset'],r['shift']) for r in rs})==32
        scores=[mean([r['gold_probability'] for r in rs if r['subset']==i]) for i in range(16)]
        sizes={r['subset']:len(r['history_indices']) for r in rs}
        selected=max(range(16),key=lambda i:(scores[i],-sizes[i],-i));assert selected==d['selected']
        assert max(abs(x-y) for x,y in zip(scores,d['fit_probabilities']))<1e-7
        assert all(r['subset']==selected for r in held if r['id']==d['id'] and r['view']=='selected')
    views={name:[r for r in held if r['view']==name] for name in ('selected','recent','all')}
    for rs in views.values():assert len({(r['id'],r['shift']) for r in rs})==628
    result=dict(completion=completion,scope='314 training questions/32 sources; gold-assisted cross-rotation diagnostic, not automatic method',
        held={name:dict(correct=sum(r['correct'] for r in rs),n=len(rs),gold_probability=mean([r['gold_probability'] for r in rs])) for name,rs in views.items()},
        selected_minus_recent=compare(views['selected'],views['recent']),selected_minus_all=compare(views['selected'],views['all']),
        fit_choice_counts={str(i):sum(d['selected']==i for d in decisions) for i in range(16)},
        limitations=['training sources only','gold-assisted choice','rotations correlated','no wrong-video/removed-evidence causal control yet','probability gain is not correctness or calibrated confidence'])
    (a.run/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
