"""Known-outcome content-vs-format and paid-overhead counterexamples, not real answers."""
import argparse
from pathlib import Path
from selector_qa_common import PREP,ARMS,read,save
from analyze_qa import analyze

def main(out):
    tasks=read(PREP/'requests.json');labels=read(PREP/'labels_LOCAL_ONLY.json')
    current=next(t for t in tasks if t['role']=='current');rows=[]
    for t in tasks:
        for arm in ARMS:
            for shift in range(5):
                idx=labels[t['id']]
                if arm=='content_visible' and t['id']==current['id']:idx=(idx+1)%5
                selected=[0] if arm in ('content_visible','matched_latest','history_blind') else list(range(4)) if arm=='full' else []
                rows.append(dict(task_id=t['id'],arm=arm,shift=shift,text=chr(65+(idx-shift)%5)+'. explanation',
                    QA_seconds=.1,selector=dict(selector_seconds=.2 if arm in ('content_visible','history_blind','matched_latest') else 0.),
                    visual_tokens=432+108*len(selected),selected_history_indices=selected))
    result=analyze(rows,tasks,labels);group=result['groups']['regression64']
    assert group['summaries']['full']['content_correct']==320
    assert group['summaries']['full']['exact_single_letter_format']==0
    assert group['against_full']['content_visible']['added_losses']==5
    assert abs(group['against_full']['content_visible']['QA_cost_ratio_vs_full']-3.)<1e-12
    save(out,dict(status='passed_synthetic_scoring',rows=len(rows),five_added_losses_detected=True,
        content_not_penalized_for_extra_text=True,selector_overhead_counted=True,real_answers=0,GPU_jobs=0))
    print('Synthetic content/format/loss/cost controls passed; real QA=0')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
