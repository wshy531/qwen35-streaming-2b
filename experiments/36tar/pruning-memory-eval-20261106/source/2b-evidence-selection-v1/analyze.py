"""Content scoring and frozen decision gates; no formatting penalty."""
import argparse,json,re
from collections import defaultdict
from pathlib import Path
from prepare import PREP,read,save

def score(rows,tasks,labels):
    byid={t['id']:t for t in tasks};arms=read(PREP/'protocol.json')['arms']
    expected={(t['id'],a,s) for t in tasks for a in arms for s in range(5)}
    keyed={(r['task_id'],r['arm'],r['shift']):r for r in rows}
    if len(rows)!=len(keyed) or set(keyed)!=expected:raise ValueError('all 1600 unique slots required')
    scored={};groups=defaultdict(lambda:dict(correct=0,available=0,missing=0,format_single_letter=0))
    for key,row in keyed.items():
        tid,arm,shift=key;task=byid[tid];available=row['available']
        m=re.search(r'\b([A-E])\b',row.get('text','').strip().upper()) if available else None
        choice=None if m is None else (ord(m.group(1))-65+shift)%5
        correct=None if not available else choice==labels[tid]
        scored[key]=correct
        g=groups[task['role']+'|'+arm];g['missing']+=not available
        if available:
            g['available']+=1;g['correct']+=bool(correct)
            g['format_single_letter']+=bool(re.fullmatch('[A-E]',row.get('text','').strip()))
    primary=[t for t in tasks if t['role']!='official_reference'];corr=[];loss=[];old_corr=set()
    for t in primary:
        for s in range(5):
            a=scored[t['id'],'time_matched_nodes',s];b=scored[t['id'],'full',s]
            if a is True and b is False:
                corr.append((t['id'],s,t['source_name']))
                if t['role']=='old':old_corr.add(t['source_name'])
            if a is False and b is True:loss.append((t['id'],s,t['source_name']))
    total=lambda role,arm:sum(scored[t['id'],arm,s] is True for t in primary if t['role']==role for s in range(5))
    all_primary_available=all(scored[t['id'],a,s] is not None for t in primary for a in arms for s in range(5))
    gates=dict(primary_complete=all_primary_available,
        at_least_eight_corrections=len(corr)>=8,correction_sources_at_least_three=len({r[2] for r in corr})>=3,
        old_correction_sources_at_least_two=len(old_corr)>=2,
        current_not_below_lexical=total('current','time_matched_nodes')>=total('current','lexical'),
        old_not_below_full=total('old','time_matched_nodes')>=total('old','full'),
        old_beats_equal_token_time_nonmatching=total('old','time_matched_nodes')>total('old','nonmatching_nodes'))
    paired=[]
    for arm in ('full','time_matched_nodes','nonmatching_nodes','lexical'):
        both=0
        for case in sorted({t['case_id'] for t in primary}):
            for s in range(5):both+=scored[case+':late:old',arm,s] is True and scored[case+':late:current',arm,s] is True
        paired.append(dict(arm=arm,both_old_and_current_correct=both,denominator=30))
    return dict(groups=dict(groups),primary_corrections=corr,primary_new_errors=loss,
        primary_old_correction_source_count=len(old_corr),gates=gates,passed=all(gates.values()),
        next_action='prepare_one_fixed_training_experiment' if all(gates.values()) else 'close_evidence_selection_hypothesis_no_parameter_sweep',
        joint=paired,rotations_not_independent=True,official_intervals_only_proxies=True,
        automatic_retrieval_tested=False,format_not_part_of_correctness=True,formal_generalization_claim=False)

def main(run,output):
    complete=read(run/'completed.json')
    if complete['status']!='complete' or complete['reference_matches']!=640:raise ValueError('complete reference reproduction required')
    rows=[json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    save(output,score(rows,read(PREP/'inputs/requests.json'),read(PREP/'audit/labels_LOCAL_ONLY.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run,a.output)
