"""Synthetic tests of label isolation, rotations and complete subset loading."""
import json
from pathlib import Path
import torch
import fit

def main():
    torch.set_num_threads(1);torch.manual_seed(20261104)
    out=fit.ROOT/'runs/history_utility_fit_cpu_20261104_01';assert not out.exists();out.mkdir()
    run=out/'fixture';run.mkdir();questions=[];features=[];rows=[]
    for i in range(2):
        q=dict(id=f'synthetic-{i}',video_id='synthetic',answer_index=i,question='What happened first?' if i else 'What is visible now?');questions.append(q)
        features.append(dict(id=q['id'],video_id='synthetic',query=torch.randn(2048),recent=torch.randn(4096),
            history=torch.randn(4,4096),metadata=torch.rand(4,8),node_mask=torch.ones(4,dtype=torch.bool)))
        for s in range(16):
            for shift in (0,2):
                gold=(i-shift)%5;logits=torch.zeros(5);logits[gold]=s/16
                p=logits.softmax(-1)
                rows.append(dict(id=q['id'],video_id='synthetic',phase='fit',subset=s,shift=shift,gold_index=gold,
                    option_logits=logits.tolist(),gold_probability=float(p[gold]),correct=int(logits.argmax())==gold,
                    visual_tokens=432+108*len(fit.subsets(4)[s])))
    manifest=[dict(video_id='synthetic',sha256='synthetic-source-only',questions=questions)]
    torch.save(dict(rows=features,source_sha256='synthetic-source-only'),run/'features-synthetic.pt')
    fit.save(run/'completed.json',dict(status='completed'))
    def write(rs):(run/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs),encoding='utf-8')
    write(rows);a=fit.load([run],manifest)
    # Held-rotation annotations are excluded from regression targets.
    write(rows+[dict(id='synthetic-0',phase='confirmation',arbitrary_label=999)])
    b=fit.load([run],manifest);assert torch.equal(a['target'],b['target'])
    assert torch.equal(a['target'][:,0],torch.zeros(2)) and bool((a['target'][:,1:]>0).all())
    write(rows[:-1])
    try:fit.load([run],manifest)
    except AssertionError:missing_rejected=True
    else:raise AssertionError('missing subset accepted')
    bad=[dict(r) for r in rows];bad[0]['gold_index']=(bad[0]['gold_index']+1)%5;write(bad)
    try:fit.load([run],manifest)
    except AssertionError:wrong_gold_rejected=True
    else:raise AssertionError('wrong rotated gold accepted')
    write(rows)
    model=fit.HistoryUtilityHead();report,paired=fit.evaluate(model,a,False)
    assert len(paired)==28 and report['metrics']['head']['answers']==4
    # Minimal synthetic optimization uses the actual regression objective.
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    prediction=model(*a['batch']);loss=fit.objective(prediction,a['target'],torch.ones_like(prediction,dtype=torch.bool),a['weights'])
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
    result=dict(status='cpu_passed',synthetic_questions=2,synthetic_combinations=64,
        held_labels_excluded=True,missing_subset_rejected=missing_rejected,rotated_gold_rejected=wrong_gold_rejected,
        NULL_zero=True,paired_controls=True,synthetic_update=True,actual_training_questions=0)
    fit.save(out/'completed.json',result);print(result)
if __name__=='__main__':main()
