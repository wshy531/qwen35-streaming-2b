"""Uniquely named checks; imported project paths also contain check.py."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from experiment import torch, rank_loss, previous


def run_checks():
    pred=torch.tensor([[0.,0.,0.,0.]],requires_grad=True)
    target=torch.tensor([[0.,.3,-.2,1e-6]])
    valid=torch.tensor([[True,True,True,False]])
    loss=rank_loss(pred,target,valid);loss.backward()
    assert pred.grad[0,1]<0 and pred.grad[0,2]>0 and pred.grad[0,3]==0
    assert abs(float(pred.grad.sum()))<1e-6
    good=target.clone();bad=-target
    assert rank_loss(good,target,valid)<rank_loss(bad,target,valid)
    torch.testing.assert_close(rank_loss(pred.detach(),target,valid),rank_loss(pred.detach()+12,target,valid))
    perm=torch.tensor([2,0,3,1])
    torch.testing.assert_close(rank_loss(good,target,valid),rank_loss(good[:,perm],target[:,perm],valid[:,perm]))
    invalid=good.clone();invalid[0,3]=500
    torch.testing.assert_close(rank_loss(good,target,valid),rank_loss(invalid,target,valid))
    empty=torch.tensor([[0.,1.,2.]],requires_grad=True)
    zero=rank_loss(empty,torch.zeros_like(empty),torch.tensor([[True,False,False]]))
    zero.backward();assert zero==0 and bool((empty.grad==0).all())
    # A batch with one informative and one uninformative row keeps denominator 2.
    v2=torch.cat([valid,torch.tensor([[True,False,False,False]])])
    p2=torch.cat([good,good]);t2=torch.cat([target,torch.zeros_like(target)])
    torch.testing.assert_close(rank_loss(p2,t2,v2),rank_loss(good,target,valid)/2)
    torch.manual_seed(20261104);head=previous.Head();b=2
    x=[torch.randn(b,3584),torch.randn(b,7168),torch.randn(b,4,7168),torch.randn(b,4,8),torch.ones(b,4,dtype=torch.bool),torch.zeros(b,3,4,dtype=torch.bool)]
    x[-1][:,1,0]=True;x[-1][:,2,:]=True
    a=head(*x,blind_history=True);changed=[v.clone() for v in x];changed[2].normal_(100,20)
    torch.testing.assert_close(a,head(*changed,blind_history=True),rtol=0,atol=0)
    assert bool((a[:,0]==0).all())
    assert sum(v.numel() for v in head.parameters())==713537
    return dict(direction=True,null_compared=True,score_shift_invariant=True,candidate_permutation_invariant=True,
                invalid_excluded=True,no_information_zero_loss_gradient=True,batch_denominator_preserved=True,
                blind_history_independent=True,parameters=713537)


if __name__=='__main__':print(run_checks())
