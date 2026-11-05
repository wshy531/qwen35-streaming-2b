import json
from pathlib import Path
import torch
from utility_head import HistoryUtilityHead,subsets,probability_gain,objective,choose

def main():
    torch.set_num_threads(1);torch.manual_seed(20261104)
    assert [len(subsets(n)) for n in range(5)]==[1,2,4,8,16]
    # Rotate both logits and gold labels: utility must be invariant.
    logits=torch.randn(2,16,1,5);gold=torch.tensor([[2],[4]])
    rotated=torch.roll(logits,shifts=-2,dims=-1)
    both=torch.cat((logits,rotated),2);targets=torch.cat((gold,(gold-2)%5),1)
    u,details=probability_gain(both,targets)
    torch.testing.assert_close(details['per_rotation_gain'][:,:,0],details['per_rotation_gain'][:,:,1])
    assert torch.equal(u[:,0],torch.zeros(2))
    choices=subsets(4);mask=torch.zeros(2,16,4,dtype=torch.bool)
    for i,ids in enumerate(choices):mask[:,i,list(ids)]=True
    batch=[torch.randn(2,2048),torch.randn(2,4096),torch.randn(2,4,4096),torch.randn(2,4,8),torch.ones(2,4,dtype=torch.bool),mask]
    model=HistoryUtilityHead();opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    p=model(*batch);assert torch.isfinite(p).all() and torch.equal(p[:,0],torch.zeros(2))
    loss=objective(p,u,torch.ones_like(u,dtype=torch.bool),torch.ones(2));loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
    before={n:x.clone() for n,x in model.named_parameters()};opt.step()
    assert any(not torch.equal(before[n],x) for n,x in model.named_parameters())
    altered=[x.clone() for x in batch];altered[2]=torch.randn_like(batch[2])*100
    torch.testing.assert_close(model(*batch,blind_history=True),model(*altered,blind_history=True),atol=0,rtol=0)
    assert choices[choose(torch.zeros(16),choices)]==()
    null=[batch[0],batch[1],batch[2],batch[3],torch.zeros(2,4,dtype=torch.bool),torch.zeros(2,1,4,dtype=torch.bool)]
    assert torch.equal(model(*null),torch.zeros(2,1))
    out=dict(status='cpu_mechanism_passed',parameters=sum(x.numel() for x in model.parameters()),subset_counts=[1,2,4,8,16],
             permutation_target_invariance=True,NULL_zero=True,blind_history_invariance=True,gradient_and_update=True,
             actual_training_examples=0,real_model_answers=0,GPU=0,capability_claim=False)
    root=Path(__file__).resolve().parents[2]/'runs/history_utility_prepare_20261104_01';root.mkdir(exist_ok=True)
    (root/'CPU_check.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(out)
if __name__=='__main__':main()
