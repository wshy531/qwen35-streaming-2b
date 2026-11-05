"""Complete fixed endpoint comparisons, including failed updates and all data."""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-training-order-main-v2'))
from analyze import metrics, paired, lines, read

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    completion=read(a.run/'completed.json');assert completion['status']=='completed'
    updates=lines(a.run/'updates.jsonl');seen=[q['id'] for u in updates for q in u['questions']]
    assert len(updates)==270 and len(seen)==len(set(seen))==2351
    assert completion['accepted']+completion['rejected']==270
    for u in updates:
        assert u['accepted']==(u['post_ce']<=u['pre_ce']+u['tolerance'])
        assert {q['id'] for q in u['questions']}=={q['id'] for q in u['pre_questions']}=={q['id'] for q in u['post_questions']}
    answers=lines(a.run/'answers.jsonl');guard=[r for r in answers if r['stage']=='final'];assert len(guard)==1774
    seq=lines(ROOT/'runs/training-order-main-220483/answers.jsonl');inter=lines(ROOT/'runs/training-order-main-220484/answers.jsonl');old=lines(ROOT/'runs/training-order-reference-220485/answers.jsonl')
    methods={'initial':[r for r in seq if r['stage']=='initial'],'old_CE':old,'sequential':[r for r in seq if r['stage']=='final'],'interleaved':[r for r in inter if r['stage']=='final'],'guarded':guard}
    tables={};comparisons={};history={}
    for role,view in [('old_development','full_short'),('long_development','LS'),('long_development','recent8')]:
        key=role+'/'+view;selected={name:[r for r in rs if r['role']==role and r['view']==view] for name,rs in methods.items()}
        tables[key]={name:metrics(rs) for name,rs in selected.items()}
        comparisons[key]={name:paired(rs,selected['guarded']) for name,rs in selected.items() if name!='guarded'}
    for name,rs in methods.items():history[name]=paired([r for r in rs if r['view']=='recent8'],[r for r in rs if r['view']=='LS'])
    training={}
    for role in ('short_train','long_train'):
        rs=[u for u in updates if u['role']==role];accepted=[u for u in rs if u['accepted']];rejected=[u for u in rs if not u['accepted']]
        training[role]=dict(proposals=len(rs),questions=sum(len(u['questions']) for u in rs),accepted=len(accepted),rejected=len(rejected),
            accepted_with_changed_selection=sum(u['changed_groups']>0 for u in accepted),
            mean_accepted_same_source_ce_delta=sum(u['post_ce']-u['pre_ce'] for u in accepted)/max(1,len(accepted)),
            mean_rejected_same_source_ce_delta=sum(u['post_ce']-u['pre_ce'] for u in rejected)/max(1,len(rejected)),
            note='local batch acceptance is training-only; not global loss or generalization guarantee')
    result=dict(completion=completion,metrics=tables,paired_guard_minus_reference=comparisons,history_vs_recent=history,training=training,
        limitations=['one seed and known development','same proposals/data but fewer accepted optimizer updates','not proof of unbiased gradient or global descent','no independent new benchmark or open-answer judge'])
    (a.run/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
