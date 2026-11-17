"""Random tiny QA/bank/recompute integration, not real training evidence."""
import sys,json,hashlib,copy
from pathlib import Path
from residual import ROOT,ContentResidual,encode,torch
sys.path.insert(0,str(ROOT/'experiments/7b-trained-prune-completion-v1'))
import training
from transformers import AutoProcessor
from native_cpu import make_tiny
torch.set_num_threads(1);torch.manual_seed(7113)
model=make_tiny(real_tokens=True).float()
assets=ROOT/'runs/7b_transfer_prepare_20261107_01/processor_assets'
if not assets.exists():assets=ROOT/'models/Qwen2.5-VL-7B-Instruct'
p=AutoProcessor.from_pretrained(assets,local_files_only=True)
panel=json.loads((ROOT/'runs/training_order_main_prepare_20261104_02/panel_LOCAL_ONLY.json').read_text(encoding='utf8'))
scorer=ContentResidual(32);opt=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
original=training.encode;reports=[]
try:
    # This process only; old source file and all saved experiments remain unchanged.
    training.encode=encode
    for role in ('short_train','long_train'):
        vid=next(v for v in panel['schedules']['sequential'] if panel['groups'][v]['role']==role)
        pairs=[(2.*i,2.*i+1,training.pair_inputs(p,torch.randint(0,256,(2,280,392,3),dtype=torch.uint8).numpy())) for i in range(9)]
        questions=panel['groups'][vid]['questions'][:2]
        stats=training.guarded_update(model,p,scorer,opt,pairs,questions,role=='short_train')
        assert stats['gradient_l1']>0
        reports.append(dict(role=role,questions=len(questions),**stats))
    from qa_gradient_check import equal
    assert opt.state_dict()['state']
    saved_parameters=copy.deepcopy(scorer.state_dict());saved_adam=copy.deepcopy(opt.state_dict())
    original_loss=training.source_loss;calls=[0]
    def reject_post(*args,**kwargs):
        calls[0]+=1
        real=original_loss(*args,**kwargs)
        return real+1000. if calls[0]==2 else real
    try:
        training.source_loss=reject_post
        rejected=training.guarded_update(model,p,scorer,opt,pairs,questions,False)
        assert not rejected['accepted'] and calls[0]==2
        assert equal(saved_parameters,scorer.state_dict()) and equal(saved_adam,opt.state_dict())
    finally: training.source_loss=original_loss
finally: training.encode=original
assert training.encode is original
out=ROOT/'runs/7b_content_residual_cpu_qa_20261113_02';out.mkdir(exist_ok=False)
report=dict(cpu_synthetic=True,GPU=0,base_updates=0,steps=3,rows=reports,
    forced_post_loss_failure_test=dict(accepted=rejected['accepted'],nonempty_adam_and_parameters_restored=True),
    limitation='Tiny random base and synthetic pixels; tests QA gradient, LS replay and guarded execution, not model capability',
    code_sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
(out/'check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
