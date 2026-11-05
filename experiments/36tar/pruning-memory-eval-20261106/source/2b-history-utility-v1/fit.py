"""CPU utility regression, activated only after complete real-label collection.

Requires the FULL frozen training/development manifests. Never treats the
32-source diagnostic as a completed 1114-question training experiment.
"""
import argparse,json,math,sys
from collections import Counter
from pathlib import Path
import torch
from utility_head import HistoryUtilityHead,subsets,probability_gain,objective,choose
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/history_utility_prepare_20261104_01'
SEED=20261104
sys.path.insert(0,str(ROOT/'experiments/2b-temporal-router-v1'))
from temporal_policy import lexical_route

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def load(directories,manifest):
    sources={g['video_id']:g for g in manifest};expected={q['id']:q for g in manifest for q in g['questions']}
    features={};answers={}
    for directory in directories:
        complete=read(directory/'completed.json');assert complete['status']=='completed',directory
        for p in directory.glob('features-*.pt'):
            item=torch.load(p,map_location='cpu',weights_only=True)
            for row in item['rows']:
                qid=row['id'];assert qid in expected and qid not in features,qid
                assert row['video_id']==expected[qid]['video_id']
                assert item['source_sha256']==sources[row['video_id']]['sha256']
                assert row['query'].shape==(2048,) and row['recent'].shape==(4096,)
                assert row['history'].shape==(4,4096) and row['metadata'].shape==(4,8)
                assert row['node_mask'].dtype==torch.bool and bool(row['node_mask'].all())
                features[qid]=row
        for s in (directory/'answers.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(s)
            if row['phase']!='fit':continue
            key=(row['id'],row['subset'],row['shift'])
            assert key not in answers and key[0] in expected and key[2] in (0,2)
            assert row['gold_index']==(expected[key[0]]['answer_index']-key[2])%5
            assert row['video_id']==expected[key[0]]['video_id']
            answers[key]=row
    assert set(features)==set(expected),dict(found=len(features),required=len(expected))
    assert len(answers)==len(expected)*32
    ids=sorted(expected);fields=('query','recent','history','metadata','node_mask')
    batch=[torch.stack([features[q][f] for q in ids]) for f in fields]
    mask=torch.zeros(len(ids),16,4,dtype=torch.bool)
    choices=subsets(4)
    for i,s in enumerate(choices):mask[:,i,list(s)]=True
    batch.append(mask)
    logits=torch.tensor([[[answers[(q,s,r)]['option_logits'] for r in (0,2)] for s in range(16)] for q in ids])
    gold=torch.tensor([[answers[(q,0,r)]['gold_index'] for r in (0,2)] for q in ids])
    targets,_=probability_gain(logits,gold)
    counts=Counter(expected[q]['video_id'] for q in ids)
    weights=torch.tensor([len(ids)/(len(counts)*counts[expected[q]['video_id']]) for q in ids])
    assert abs(float(weights.mean())-1)<1e-6
    assert all(torch.isfinite(x).all() for x in batch if x.is_floating_point())
    return dict(ids=ids,batch=batch,target=targets,weights=weights,answers=answers,choices=choices,
        source_ids=[expected[q]['video_id'] for q in ids],question_text=[expected[q]['question'] for q in ids])

@torch.no_grad()
def evaluate(model,data,blind):
    predictions=[]
    for start in range(0,len(data['ids']),32):
        predictions.append(model(*[x[start:start+32] for x in data['batch']],blind_history=blind))
    prediction=torch.cat(predictions)
    choices=data['choices'];indices=[choose(p,choices) for p in prediction]
    rows=[]
    for n,qid in enumerate(data['ids']):
        k=len(choices[indices[n]])
        fixed={'recent':0,'all':15,'nearest2':choices.index((2,3)),
            'head':indices[n],'matched_count_nearest':choices.index(tuple(range(4-k,4))),
            'same_head_binary':15 if float(prediction[n,15])>0 else 0,
            'lexical_rule':15 if lexical_route(data['question_text'][n])['route']=='full' else 0}
        for name,index in fixed.items():
            for shift in (0,2):
                r=data['answers'][(qid,index,shift)]
                rows.append(dict(id=qid,video_id=r['video_id'],shift=shift,view=name,subset=index,
                    correct=r['correct'],gold_probability=r['gold_probability'],visual_tokens=r['visual_tokens'],
                    predicted_gain=float(prediction[n,index]),measured_gain=float(data['target'][n,index])))
    stats={}
    for name in ('recent','all','nearest2','head','matched_count_nearest','same_head_binary','lexical_rule'):
        rs=[r for r in rows if r['view']==name]
        stats[name]=dict(correct=sum(r['correct'] for r in rs),answers=len(rs),
            mean_gold_probability=sum(r['gold_probability'] for r in rs)/len(rs),
            mean_visual_tokens=sum(r['visual_tokens'] for r in rs)/len(rs))
    return dict(metrics=stats,huber=float(objective(prediction,data['target'],torch.ones_like(prediction,dtype=torch.bool),data['weights'])),
        counts_by_node_number=dict(Counter(len(choices[i]) for i in indices))),rows

def fit(train,development,out,blind):
    torch.manual_seed(SEED);model=HistoryUtilityHead();assert sum(p.numel() for p in model.parameters())==418625
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    generator=torch.Generator().manual_seed(SEED);history=[]
    for epoch in range(1,21):
        order=torch.randperm(len(train['ids']),generator=generator);loss_sum=0.;examples=0
        for start in range(0,len(order),16):
            idx=order[start:start+16];optimizer.zero_grad(set_to_none=True)
            prediction=model(*[x[idx] for x in train['batch']],blind_history=blind)
            loss=objective(prediction,train['target'][idx],torch.ones_like(prediction,dtype=torch.bool),train['weights'][idx])
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach())*len(idx);examples+=len(idx)
        # Log training only; no development-selected epochs or thresholds.
        history.append(dict(epoch=epoch,training_huber=loss_sum/examples,examples=examples,last_grad_norm=float(norm)))
    report,rows=evaluate(model,development,blind)
    name='blind_history' if blind else 'content_visible'
    torch.save(dict(state_dict=model.state_dict(),seed=SEED,epoch=20,train_questions=len(train['ids']),
        blind_history=blind,query_input='recent plus question text, no options/answers'),out/f'{name}.pt')
    save(out/f'{name}_training.json',history);save(out/f'{name}_development.json',report)
    with (out/f'{name}_paired.jsonl').open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-run',type=Path,action='append',required=True)
    ap.add_argument('--dev-run',type=Path,action='append',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists() and a.out.resolve().is_relative_to((ROOT/'runs').resolve())
    torch.set_num_threads(1)
    tr=read(PREP/'train_manifest_LOCAL_ONLY.json');dv=read(PREP/'development_manifest_LOCAL_ONLY.json')
    assert len(tr)==120 and sum(len(g['questions']) for g in tr)==1114
    assert len(dv)==30 and sum(len(g['questions']) for g in dv)==271
    for key in ('video_id','sha256'):assert not ({g[key] for g in tr}&{g[key] for g in dv})
    train=load(a.train_run,tr);development=load(a.dev_run,dv);assert not (set(train['ids'])&set(development['ids']))
    a.out.mkdir();results={}
    for blind in (False,True):results['blind' if blind else 'visible']=fit(train,development,a.out,blind)
    save(a.out/'completed.json',dict(status='completed',train_questions=1114,development_questions=271,epochs=20,
        source_separated=True,development_previously_opened=True,base_updates=0,GPU=0,results=results))
    print(json.dumps(results))
if __name__=='__main__':main()
