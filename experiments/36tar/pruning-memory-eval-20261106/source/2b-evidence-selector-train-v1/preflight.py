"""CPU gradient, invariance, supervision and bounded-selection contracts."""
import os,json,sys,torch
from pathlib import Path
from prepare_data import ROOT,PREP,read,save,sha
from selector_model import EvidenceSelector,objective,select,metadata

def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' or output.exists():raise RuntimeError('CPU only/new result required')
    torch.set_num_threads(1);torch.manual_seed(20261029)
    model=EvidenceSelector();batch=dict(query=torch.randn(8,2048),recent=torch.randn(8,4096),
        history=torch.randn(8,4,4096),metadata=torch.rand(8,4,8),mask=torch.ones(8,4,dtype=torch.bool),
        route_known=torch.tensor([1,1,1,1,1,1,0,0],dtype=torch.bool),needs_history=torch.tensor([0.,1.,0.,1.,0.,1.,0.,0.]),
        route_weight=torch.ones(8),positive_mask=torch.zeros(8,4,dtype=torch.bool),rank_weight=torch.ones(8))
    batch['positive_mask'][[1,3,5,6,7],0]=True
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[]
    for _ in range(12):
        model.train();optimizer.zero_grad();loss,_=objective(model,batch)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
    assert any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    model.eval();changed={**batch,'history':batch['history']+torch.randn_like(batch['history'])*3}
    args=lambda x:[x[k] for k in ('query','recent','history','metadata','mask')]
    with torch.no_grad():
        visible=model(*args(batch))[0];changed_visible=model(*args(changed))[0]
        blind=model(*args(batch),blind_history=True)[0];changed_blind=model(*args(changed),blind_history=True)[0]
    assert not torch.equal(visible,changed_visible) and torch.equal(blind,changed_blind)
    # NULL can abstain from historical reading; selection never requires filling k.
    assert select(torch.tensor([5.,1.,2.,3.,4.]),torch.tensor(-1.),[108]*4)['history_indices']==[]
    assert select(torch.tensor([0.,5.,4.,3.,2.]),torch.tensor(3.),[108]*4)['history_indices']==[0,1]
    try:metadata(dict(begin=5,end=15,tokens=108,exact_pair=True),10,8)
    except ValueError:pass
    else:raise AssertionError('future history admitted')
    sup=read(PREP/'supervision_LOCAL_ONLY.json');train=[v for v in sup.values() if v['role']=='train']
    assert len(train)==195 and sum(v['needs_history'] for v in train)==46
    assert all(v['positive_history_indices'] for v in train if v['needs_history'])
    long=read(PREP/'long_supervision_LOCAL_ONLY.json');assert all(not v['route_known'] for v in long.values())
    rows=read(PREP/'short_requests.json')+read(PREP/'long_requests.json')
    forbidden={'reference_answer','correct_index','reference_evidence_ids','positive_history_indices','intervals','needs_history'}
    assert all(not forbidden&set(r) for r in rows)
    save(output,dict(status='passed_CPU_selector_mechanism',parameters=sum(p.numel() for p in model.parameters()),
        synthetic_training_updates=12,initial_loss=losses[0],final_loss=losses[-1],real_training_examples_used=0,
        finite_all_parameter_gradients=True,visible_history_perturbation_changes_scores=True,
        blind_history_perturbation_no_effect=True,null_and_node_cap_checked=True,
        future_history_rejected=True,short_train_labels=195,long_rows_never_train_route=True,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')},
        GPU_jobs=0,pretrained_loads=0,capability_claim=False))
    print(dict(status='passed',parameters=sum(p.numel() for p in model.parameters()),synthetic_updates=12,real_training=0))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
