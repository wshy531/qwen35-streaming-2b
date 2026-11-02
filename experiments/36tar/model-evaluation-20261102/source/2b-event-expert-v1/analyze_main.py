"""Content-only canonical action classification; paired gains and video/text controls."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from common import PREP,LABELS,read,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run);completed=read(run/'completed.json')
    assert completed['status']=='fixed_main_complete' and completed['total_classifications']==2104
    source={r['id']:r for r in read(PREP/'media_frames_LOCAL_ONLY.json')}
    arms=('original','task_only','expert_aligned','expert_shuffled');all_rows=[];lookup={}
    for arm in arms:
        rows=read(run/(arm+'_predictions.json'));assert len(rows)==526
        for r in rows:
            assert r['arm']==arm and r['label']==source[r['id']]['label'] and r['split']==source[r['id']]['split']
            assert r['predicted']==LABELS[max(range(7),key=lambda k:r['scores'][k])]
            assert r['mode'] in ('video','text_only') and (r['mode']=='video' or r['split']=='test')
            key=(arm,r['mode'],r['id']);assert key not in lookup;lookup[key]=r
            r['correct']=r['predicted']==r['label'];all_rows.append(r)
        for mode,split in (('video','validation'),('video','test'),('text_only','test')):
            expected={i for i,r in source.items() if r['split']==split}
            actual={r['id'] for r in rows if r['mode']==mode and r['split']==split}
            assert actual==expected
    def summary(rows):
        by_class={};by_video=defaultdict(list);by_person=defaultdict(list)
        for label in LABELS:
            x=[r for r in rows if r['label']==label]
            by_class[label]=dict(n=len(x),correct=sum(r['correct'] for r in x),
                accuracy=sum(r['correct'] for r in x)/len(x) if x else None)
        for r in rows:
            by_video[source[r['id']]['video_id']].append(r['correct'])
            by_person[source[r['id']]['participant_id']].append(r['correct'])
        return dict(n=len(rows),correct=sum(r['correct'] for r in rows),
            accuracy=sum(r['correct'] for r in rows)/len(rows),per_class=by_class,
            macro_over_observed_classes=sum(v['accuracy'] for v in by_class.values() if v['n'])/sum(v['n']>0 for v in by_class.values()),
            per_participant={p:dict(n=len(v),correct=sum(v)) for p,v in by_person.items()},
            per_video={p:dict(n=len(v),correct=sum(v)) for p,v in by_video.items()},
            predicted_counts=dict(Counter(r['predicted'] for r in rows)))
    results={arm:{mode+'_'+split:summary([r for r in all_rows if r['arm']==arm and r['mode']==mode and r['split']==split])
        for mode,split in (('video','validation'),('video','test'),('text_only','test'))} for arm in arms}
    def paired(a,b,mode='video'):
        ids=[i for i,r in source.items() if r['split']=='test'];gains=[];losses=[]
        for i in ids:
            x=lookup[(a,mode,i)]['correct'];y=lookup[(b,mode,i)]['correct']
            if x and not y:gains.append(i)
            if y and not x:losses.append(i)
        return dict(n=len(ids),gains=len(gains),losses=len(losses),net=len(gains)-len(losses),gain_ids=gains,loss_ids=losses)
    contrasts={'expert_vs_task':paired('expert_aligned','task_only'),
        'expert_vs_shuffled':paired('expert_aligned','expert_shuffled'),
        'task_vs_original':paired('task_only','original')}
    visual={}
    for arm in arms:
        ids=[i for i,r in source.items() if r['split']=='test']
        gain=sum(lookup[(arm,'video',i)]['correct'] and not lookup[(arm,'text_only',i)]['correct'] for i in ids)
        loss=sum(lookup[(arm,'text_only',i)]['correct'] and not lookup[(arm,'video',i)]['correct'] for i in ids)
        visual[arm]=dict(video_over_text_gains=gain,video_over_text_losses=loss,net=gain-loss)
    save(run/'analysis.json',dict(status='complete_content_analysis',results=results,contrasts=contrasts,
        visual_vs_text=visual,format_not_scored=True,missing_hold_not_scored_as_zero=True,
        independent_test_participants=3,dependent_rows_not_independent_trials=True,
        single_seed_one_epoch=True,long_stream_memory_improvement_not_established=True))
    print({a:results[a]['video_test']['correct'] for a in arms})


if __name__=='__main__':main()
