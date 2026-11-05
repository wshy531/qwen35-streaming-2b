"""Content quality, added losses and selection-inclusive QA cost; never retrain."""
import argparse,json,math,re,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from selector_qa_common import ROOT,PREP,ARMS,read,save

def analyze(rows,tasks,labels):
    bytask={t['id']:t for t in tasks}
    index={(r['task_id'],r['arm'],r['shift']):r for r in rows}
    expected={(t['id'],a,s) for t in tasks for a in ARMS for s in range(5)}
    if len(rows)!=len(index) or set(index)!=expected:raise ValueError('Incomplete/duplicate/foreign QA rows')
    def choice(r):
        m=re.search(r'\b([A-E])\b',r['text'].strip().upper())
        return None if m is None else (ord(m.group(1))-65+r['shift'])%5
    correct={k:choice(r)==labels[k[0]] for k,r in index.items()}
    groups=defaultdict(list)
    for t in tasks:
        groups[t['evaluation_group']].append(t['id'])
        if t['evaluation_group']=='regression64':groups['regression64/'+t['role']].append(t['id'])
        else:groups['long36/category/'+t['category']].append(t['id'])
    result={}
    for group,ids in groups.items():
        summaries={}
        for arm in ARMS:
            part=[index[(tid,arm,s)] for tid in ids for s in range(5)]
            qa=np.array([r['QA_seconds'] for r in part]);sel=np.array([r['selector']['selector_seconds'] for r in part])
            total=qa+sel
            summaries[arm]=dict(content_correct=sum(correct[(tid,arm,s)] for tid in ids for s in range(5)),
                denominator=len(part),question_times=len(ids),source_files=len({bytask[tid]['source_name'] for tid in ids}),
                exact_single_letter_format=sum(r['text'].strip() in list('ABCDE') for r in part),
                unparsed=sum(choice(r) is None for r in part),
                mean_visual_tokens=float(np.mean([r['visual_tokens'] for r in part])),
                history_read_questions=sum(bool(index[(tid,arm,0)]['selected_history_indices']) for tid in ids),
                QA_only_mean_seconds=float(qa.mean()),selector_mean_seconds=float(sel.mean()),
                QA_with_selection_mean_seconds=float(total.mean()),QA_with_selection_median_seconds=float(np.median(total)),
                QA_with_selection_p95_seconds=float(np.quantile(total,.95)))
        comparisons={}
        for arm in ARMS:
            if arm=='full':continue
            gained=[(tid,s) for tid in ids for s in range(5) if correct[(tid,arm,s)] and not correct[(tid,'full',s)]]
            lost=[(tid,s) for tid in ids for s in range(5) if not correct[(tid,arm,s)] and correct[(tid,'full',s)]]
            sources=sorted({bytask[tid]['source_name'] for tid in ids});blocks=[]
            for src in sources:
                tids=[tid for tid in ids if bytask[tid]['source_name']==src]
                blocks.append((sum(int(correct[(tid,arm,s)])-int(correct[(tid,'full',s)]) for tid in tids for s in range(5)),len(tids)*5))
            rng=np.random.default_rng(20261030);effects=[]
            for _ in range(2000):
                draw=rng.integers(0,len(blocks),len(blocks));effects.append(sum(blocks[i][0] for i in draw)/sum(blocks[i][1] for i in draw))
            comparisons[arm]=dict(corrections=len(gained),added_losses=len(lost),net=len(gained)-len(lost),
                source_cluster_bootstrap_difference_CI95=[float(x) for x in np.quantile(effects,[.025,.975])],
                QA_cost_ratio_vs_full=summaries[arm]['QA_with_selection_mean_seconds']/summaries['full']['QA_with_selection_mean_seconds'],
                gained_rows=gained,lost_rows=lost)
        result[group]=dict(summaries=summaries,against_full=comparisons)
    paired={}
    cases=sorted({t['case_id'] for t in tasks if 'case_id' in t})
    for arm in ARMS:
        n=0;both=0
        for case in cases:
            olds=[t for t in tasks if t.get('case_id')==case and t['role']=='old']
            currents=[t for t in tasks if t.get('case_id')==case and t['role']=='current' and t.get('stage')=='late']
            if not olds or not currents:continue
            for s in range(5):
                n+=1;both+=int(correct[(olds[0]['id'],arm,s)] and correct[(currents[0]['id'],arm,s)])
        paired[arm]=dict(both_current_and_old_correct=both,denominator=n)
    return dict(groups=result,same_time_old_and_current=paired,unit='source/question; option permutations are correlated',
        content_and_format_separate=True,standalone_end_to_end_speed_claim=False,
        automatic_candidate_adoption=False,threshold_or_epoch_search=False)

def main(out):
    done=read(out/'completed.json');protocol=read(PREP/'protocol.json')
    if done['status']!='complete' or done['reference_matches']!=640 or done['formal_answers']!=3000:
        raise ValueError('Only analyze completed qualified run')
    rows=[json.loads(s) for s in (out/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    if sum(len(r['generated_ids']) for r in rows)>protocol['maximum_generation_tokens']:raise ValueError('Token bound')
    for r in rows:
        if abs(r['QA_with_selection_seconds']-r['QA_seconds']-r['selector']['selector_seconds'])>1e-9:raise ValueError('Unpaid selector overhead')
        if r['selected_history_indices']!=r['selector']['selected_history_indices']:raise ValueError('Selection/reader mismatch')
    result=analyze(rows,read(PREP/'requests.json'),read(PREP/'labels_LOCAL_ONLY.json'))
    save(out/'analysis.json',result)
    print(json.dumps({k:{a:v['content_correct'] for a,v in g['summaries'].items()} for k,g in result['groups'].items()},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
