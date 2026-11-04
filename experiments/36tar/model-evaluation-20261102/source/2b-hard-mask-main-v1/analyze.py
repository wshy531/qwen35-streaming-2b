"""Fixed content scoring; old baselines explicitly marked as reused evidence."""
import argparse,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];P=R/'runs/hard_mask_main_prepare_20261102_01'
O=R/'runs/learned_prune_main_prepare_20261101_01'
ARMS=('full','fixed12','heuristic11','mask_initial','mask_ce','mask_ce_kl')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    done=read(a.run/'completed.json');assert done['status']=='complete'
    qa=lines(a.run/'answers.jsonl');base=read(P/'baseline_answers.json');regress=read(a.run/'baseline_regression.json')
    assert len(qa)==3912 and len(regress)==72 and all(r['matched'] for r in regress)
    questions={q['id']:q for q in read(O/'eval.json')};rows=qa+base
    index={(r['id'],r['arm'],r['shift']):r for r in rows}
    assert len(rows)==len(index)==7824
    assert set(index)=={(q,arm,s) for q in questions for arm in ARMS for s in (0,2)}
    for r in rows:
        q=questions[r['id']]
        assert r['video_id']==q['video_id'] and r['role']==q['role']
        assert r['correct']==(r['original_selected_index']==q['answer_index'])
    output={}
    for role in ('development','confirmation'):
        ids=[q for q in questions if questions[q]['role']==role];videos=sorted({questions[q]['video_id'] for q in ids})
        table={}
        for arm in ARMS:
            rr=[index[q,arm,s] for q in ids for s in (0,2)]
            table[arm]=dict(primary_correct=sum(index[q,arm,0]['correct'] for q in ids),questions=len(ids),
                two_rotations_correct=sum(r['correct'] for r in rr),two_rotations_answers=len(rr),
                both_rotations_correct=sum(all(index[q,arm,s]['correct'] for s in (0,2)) for q in ids),
                same_answer_content=sum(index[q,arm,0]['original_selected_index'] is not None and index[q,arm,0]['original_selected_index']==index[q,arm,2]['original_selected_index'] for q in ids),
                only_letter_format=sum(r['format_only_letter'] for r in rr),
                provenance='reused_220024_with_72_reference_checks' if arm in ARMS[:3] else 'new_repair_run',
                by_type={typ:dict(correct=sum(index[q,arm,0]['correct'] for q in ids if questions[q]['type']==typ),questions=sum(questions[q]['type']==typ for q in ids)) for typ in sorted({questions[q]['type'] for q in ids})})
        comparisons={};counts=np.array([sum(questions[q]['video_id']==v for q in ids) for v in videos])
        for candidate in ('mask_ce','mask_ce_kl'):
            comparisons[candidate]={}
            for reference in ARMS:
                if reference==candidate:continue
                gains=[q for q in ids if index[q,candidate,0]['correct'] and not index[q,reference,0]['correct']]
                losses=[q for q in ids if not index[q,candidate,0]['correct'] and index[q,reference,0]['correct']]
                delta=np.array([sum(int(index[q,candidate,0]['correct'])-int(index[q,reference,0]['correct']) for q in ids if questions[q]['video_id']==v) for v in videos])
                sample=np.random.default_rng(20261102).integers(0,len(videos),size=(2000,len(videos)))
                boot=100*delta[sample].sum(1)/counts[sample].sum(1)
                comparisons[candidate][reference]=dict(gained_ids=gains,lost_ids=losses,net=len(gains)-len(losses),
                    delta_pp=100*(len(gains)-len(losses))/len(ids),source_bootstrap_95_pp=np.quantile(boot,[.025,.975]).tolist())
        output[role]=dict(methods=table,comparisons=comparisons,video_files=len(videos),
                          limitation='opened regression sources, not unseen validation; parent capture independence unknown')
    probes=lines(a.run/'training_probes.jsonl');assert len(probes)==72
    ids={q['id'] for q in read(P/'training_probes.json')}
    assert {(r['id'],r['arm']) for r in probes}=={(q,arm) for q in ids for arm in ARMS[3:]}
    train_summary={arm:dict(ce_mean=float(np.mean([r['ce'] for r in probes if r['arm']==arm])),
                           option_correct=sum(r['option_correct'] for r in probes if r['arm']==arm),
                           first_token_correct=sum(r['first_token_correct'] for r in probes if r['arm']==arm)) for arm in ARMS[3:]}
    result=dict(quality=output,training_probes=train_summary,run=done,base_source='220024',
                limitations=['two arrangements are correlated, not extra independent questions','shared decode quality run, not standalone cost','final endpoints only; no checkpoint cherry-picking'])
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({role:{arm:x['primary_correct'] for arm,x in output[role]['methods'].items()} for role in output}))
if __name__=='__main__':main()
