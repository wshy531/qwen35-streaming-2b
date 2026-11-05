"""Paired quality precheck: all failures, rotation dependence, and source-level deltas."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/streamingbench-router-v1'))
from analyze import analyze
def main(folder):
    prep=ROOT/'runs/streamingbench_router_prepare_20261101_01'
    tasks=json.loads((prep/'profile_requests.json').read_text(encoding='utf-8'))
    labels=json.loads((ROOT/'runs/streamingbench_download_prepare_20261030_01/eval_inputs/labels_LOCAL_ONLY.json').read_text(encoding='utf-8'))
    rows=[json.loads(x) for x in (folder/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    arms=['full','mid_fixed','mid_content'];result=analyze(rows,tasks,labels,arms)
    byid={q['id']:q for q in tasks};index={(r['task_id'],r['arm'],r['shift']):r for r in result['scored']}
    paired={}
    for arm,ref in [('mid_fixed','full'),('mid_content','full'),('mid_content','mid_fixed')]:
        lost=[];gained=[];retained=bothwrong=0
        for q in tasks:
            for s in range(4):
                a=index[q['id'],arm,s];b=index[q['id'],ref,s]
                if b['correct'] and not a['correct']:lost.append(dict(task_id=q['id'],shift=s))
                elif a['correct'] and not b['correct']:gained.append(dict(task_id=q['id'],shift=s))
                elif a['correct']:retained+=1
                else:bothwrong+=1
        paired[arm+'_vs_'+ref]=dict(gained=gained,lost=lost,gained_n=len(gained),lost_n=len(lost),
            net=len(gained)-len(lost),both_correct=retained,both_wrong=bothwrong,
            full_correct_retention=retained/(retained+len(lost)) if retained+len(lost) else None)
    stability={}
    for arm in arms:
        stability[arm]=dict(all_four_correct=sum(all(index[q['id'],arm,s]['correct'] for s in range(4)) for q in tasks),
            any_correct=sum(any(index[q['id'],arm,s]['correct'] for s in range(4)) for q in tasks),
            stable_content_choice=sum(len({index[q['id'],arm,s]['choice'] for s in range(4)})==1 for q in tasks),questions=40)
    result.update(paired=paired,stability=stability,unique_sources=8,unique_questions=40,
        these_eight_sources_opened_development=True,remaining_192_files_not_yet_tested=True,
        full_is_eight_pair_window_not_native_complete_video=True,
        different_bank_coverage_from_router_LS=True,shared_cost_not_independent=True)
    with (folder/'analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(scores=[g for g in result['groups'] if g['group']=='all'],
        paired={k:{f:v[f] for f in ('gained_n','lost_n','net','both_correct','both_wrong')} for k,v in paired.items()},stability=stability),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
