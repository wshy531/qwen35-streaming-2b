import sys,json,hashlib
from pathlib import Path
from residual import ROOT,ContentResidual,select,encode,torch
sys.path.insert(0,str(ROOT/'experiments/7b-prune-selector-transfer-v1'))
import vision
sys.path.insert(0,str(ROOT/'experiments/7b-memory-transfer-v1'))
from native_cpu import make_tiny

torch.set_num_threads(1);torch.manual_seed(7113)
reference=vision.CoverageScorer();scorer=ContentResidual(32);past=None
for i in range(32):
    x=torch.zeros(560,32) if i==0 else torch.randn(560,32)
    scores,past=scorer(x,past)
    assert torch.equal(select(scores),reference.choose(x,(10,14)))
model=make_tiny().float();reference=vision.CoverageScorer();past=None;maximum=0.
for i in range(3):
    pixels=torch.randn(560,1176);grid=torch.tensor([[1,20,28]])
    with torch.no_grad():
        expected,ids,_=vision.encode(model.model.visual,pixels,grid,mode='content',scorer=reference)
        actual,selected,_,newpast=encode(model.model.visual,pixels,grid,scorer,past,'gather')
    assert torch.equal(ids,selected)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    maximum=max(maximum,float((actual-expected).abs().max()))
    mask,mid,_,_=encode(model.model.visual,pixels,grid,scorer,past,'mask')
    assert torch.equal(mid,selected)
    torch.testing.assert_close(mask.detach(),actual,atol=1e-6,rtol=1e-5)
    mask.square().mean().backward();past=newpast
assert float(scorer.correction.net[-1].weight.grad.abs().sum())>0
assert all(p.grad is None for p in model.parameters())
report=dict(random_selection_pairs=32,mini_visual_pairs=3,exact_initial_ids=True,max_feature_difference=maximum,
    mask_gather_passed=True,visual_surrogate_gradient_nonzero=True,base_frozen=True,
    parameters=sum(p.numel() for p in ContentResidual(1280).parameters()),GPU=0,
    limitation='Random tiny visual objective only; real QA, bank recomputation and optimizer acceptance not yet tested',
    code_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
out=ROOT/'runs/7b_content_residual_cpu_20261113_01';out.mkdir(exist_ok=False)
(out/'check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
