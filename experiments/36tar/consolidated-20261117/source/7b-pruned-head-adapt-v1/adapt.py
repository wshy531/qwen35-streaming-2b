"""One-variable adaptation: same sources and optimizer, pruned bank supervision."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-prune-head-combination-v1'))
from combination import *
from selector import objective
HERE=ROOT/'runs/7b_pruned_head_adapt_prepare_20261111_01'

def verify_adapt():
    verify_all()
    for rel,h in read(HERE/'lock.json').items():assert sha(ROOT/rel)==h,rel

def train_head(features,scores,groups,out):
    """No evaluation inputs or labels accepted. Fixed epoch, no checkpoint search."""
    torch.set_num_threads(1)
    gold={q['id']:q['answer_index'] for g in groups for q in g['questions']}
    rows={(r['id'],r['subset'],r['shift']):r for r in scores}
    assert len(rows)==len(scores)
    train=[f for f in features if f['role']=='train']
    dev=[f for f in features if f['role']=='development']
    assert (len(train),len(dev))==(144,32)
    assert len({f['source_id'] for f in train})==72
    assert not ({f['source_id'] for f in train}&{f['source_id'] for f in dev})
    def pack(fs):
        batch=[torch.stack([f[k] for f in fs]) for k in FIELDS]
        candidates=torch.zeros(len(fs),16,4,dtype=torch.bool)
        valid=torch.zeros(len(fs),16,dtype=torch.bool);target=torch.zeros(len(fs),16)
        for j,f in enumerate(fs):
            ps=[]
            for i,choice in enumerate(f['choices']):
                candidates[j,i,choice]=True;valid[j,i]=True;vals=[]
                for shift in (0,2):
                    logits=torch.tensor(rows[f['id'],i,shift]['option_logits'])
                    vals.append(logits.softmax(-1)[(gold[f['id']]-shift)%len(logits)])
                ps.append(float(sum(vals)/2))
            target[j,:len(ps)]=torch.tensor(ps)-ps[0]
        return batch+[candidates],valid,target
    batch,valid,target=pack(train);weights=torch.ones(len(train))
    torch.manual_seed(20261104);head=Head()
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
    gen=torch.Generator().manual_seed(20261104);trajectory=[];updates=0
    with torch.enable_grad():
        for epoch in range(1,21):
            order=torch.randperm(len(train),generator=gen);losses=[]
            for start in range(0,len(order),16):
                idx=order[start:start+16];opt.zero_grad(set_to_none=True)
                pred=head(*[x[idx] for x in batch],blind_history=False)
                loss=objective(pred,target[idx],valid[idx],weights[idx]);assert torch.isfinite(loss)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                opt.step();updates+=1;losses.append(float(loss.detach()))
            trajectory.append(dict(epoch=epoch,mean_huber=sum(losses)/len(losses),last_grad_norm=float(norm)))
            save(out/'training.json',dict(trajectory=trajectory,updates=updates))
    head.eval().requires_grad_(False)
    torch.save(dict(state_dict=head.state_dict(),width=3584,epoch=20,train_questions=144,
        train_sources=72,blind_history=False,seed=20261104,bank='content70'),out/'head-visible.pt')
    assert updates==180 and sum(x.numel() for x in head.parameters())==713537
    with torch.no_grad():
        db,dv,dt=pack(dev)
        dev_loss=float(objective(head(*db),dt,dv,torch.ones(len(dev))))
    save(out/'freeze.json',dict(head_sha=sha(out/'head-visible.pt'),updates=updates,
        train_questions=144,development_questions=32,development_huber=dev_loss,
        final_epoch_fixed=20,evaluation_opened=False))
    return head
