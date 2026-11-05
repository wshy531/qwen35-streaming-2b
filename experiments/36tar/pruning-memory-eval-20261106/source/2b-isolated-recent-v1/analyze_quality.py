"""Offline content scoring; source-level uncertainty, separate familiar diagnostics."""
import argparse,random,statistics
from collections import defaultdict
from experiment import *

def analyze(out):
    done=read(out/'completed.json');assert done['status']=='complete' and done['answers']==1704
    rows=[json.loads(s) for s in (out/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    qs={q['question_id']:q for q in read(P/'requests.json')};pq={q['id']:q for q in read(P/'paired_questions.json')}
    labels={q['question_id']:q for q in read(R/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json')}
    pl=read(R/'runs/recent_detail_prepare_20261103_01/labels_LOCAL_ONLY.json')
    expected={('official',q,0,a) for q in qs for a in ARMS}|{('paired',q,s,a) for q in pq for s in range(5) for a in ARMS}
    keyed={(r['panel'],r['id'],r['shift'],r['arm']):r for r in rows}
    assert len(keyed)==len(rows)==len(expected) and set(keyed)==expected
    for (panel,key,shift,arm),r in keyed.items():
        q=qs[key] if panel=='official' else pq[key];letter=parse_choice(r['text'],len(q['options']))
        selected=None if letter is None else (ord(letter)-65+shift)%len(q['options'])
        target=ord(labels[key]['correct_letter'])-65 if panel=='official' else pl[key]['correct_index']
        r.update(original_selected_index=selected,correct=selected==target,parsed=selected is not None)
        assert r['read_state_unchanged'] and r['branch_released']
    def stats(rs):
        return dict(answers=len(rs),content_correct=sum(r['correct'] for r in rs),unparsed=sum(not r['parsed'] for r in rs),
            measured_QA_seconds=sum(r['QA_wall_with_audit_seconds'] for r in rs),
            peak_allocated_bytes=max((r['peak_allocated_bytes'] for r in rs),default=0))
    official={a:stats([keyed['official',q,0,a] for q in qs]) for a in ARMS}
    categories={c:{a:stats([keyed['official',q,0,a] for q in qs if qs[q]['question_category']==c]) for a in ARMS} for c in sorted({q['question_category'] for q in qs.values()})}
    paired={}
    for a in ARMS:
        paired[a]={role:stats([keyed['paired',q,s,a] for q in pq for s in range(5) if pq[q]['role']==role]) for role in ('current','old')}
        paired[a]['by_rotation']={str(s):{role:sum(keyed['paired',q,s,a]['correct'] for q in pq if pq[q]['role']==role) for role in ('current','old')} for s in range(5)}
        cases=sorted({q['case_id'] for q in pq.values()})
        paired[a]['same_time_both_correct']=sum(all(keyed['paired',f'{case}:late:{role}',s,a]['correct'] for role in ('current','old')) for case in cases for s in range(5))
        paired[a]['same_time_pairs']=len(cases)*5
    comparisons={}
    for base in ('recent','joint'):
        gains=[q for q in qs if keyed['official',q,0,'isolated']['correct'] and not keyed['official',q,0,base]['correct']]
        losses=[q for q in qs if keyed['official',q,0,base]['correct'] and not keyed['official',q,0,'isolated']['correct']]
        groups=defaultdict(list)
        for q in qs:groups[qs[q]['video_id']].append(int(keyed['official',q,0,'isolated']['correct'])-int(keyed['official',q,0,base]['correct']))
        source=list(groups);rng=random.Random(20261103);values=[]
        for _ in range(2000):
            selected=[groups[rng.choice(source)] for _ in source];values.append(sum(sum(x) for x in selected)/sum(len(x) for x in selected))
        values.sort()
        comparisons[base]=dict(gained=gains,lost=losses,net=len(gains)-len(losses),source_bootstrap_exploratory_95=[values[49],values[1949]],
            caveat='Opened development sources; interval does not establish unseen generalization')
    result=dict(run=done,official=official,categories=categories,paired=paired,isolated_vs=comparisons,
        timing_caveat='Shared vision, QA includes diagnostics/audits; not standalone end-to-end or edge cost',
        content_metric='Parsed selected answer content, not evidence-grounded explanation score',
        paired_caveat='18 familiar single-assistant labels from five videos; 90 rotations are not independent questions')
    save(out/'analysis.json',result)
    with (out/'scored_LOCAL_ONLY.jsonl').open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(dict(official=official,paired=paired,comparison={b:{k:v for k,v in x.items() if k not in ('gained','lost')} for b,x in comparisons.items()}),ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);analyze(p.parse_args().out)
