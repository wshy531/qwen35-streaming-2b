"""Random tiny actual vision modules and frozen linear diagnostic readout; no QA claim."""
import os,sys,json,argparse,time
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import runtime
from bias_prune import GroupScorer,biased_block,encode

def main(out):
    start=time.monotonic();env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261102)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    model.requires_grad_(False);visual=model.model.visual
    original={k:v.detach().clone() for k,v in visual.state_dict().items()}
    width=visual.config.hidden_size;grid=t.tensor([[1,18,24]]);pixels=t.randn(432,1536)
    x=t.randn(432,width);rope=visual.rot_pos_emb(grid).reshape(432,-1);emb=t.cat((rope,rope),-1);pos=(emb.cos(),emb.sin())
    ref=visual.blocks[1](x,cu_seqlens=t.tensor([0,432],dtype=t.int32),position_embeddings=pos)
    errors=[]
    for value in [0.,2.]:
        y=biased_block(visual.blocks[1],x,pos,t.full((108,),value))
        errors.append(float((y-ref).abs().max()));assert t.allclose(y,ref,atol=2e-6,rtol=2e-5)
    scorer=GroupScorer(width);before={k:v.detach().clone() for k,v in scorer.state_dict().items()}
    features,ids,logits,state=encode(visual,pixels,grid,scorer)
    assert features.shape==(54,32) and len(ids.unique())==54 and t.all(ids[1:]>ids[:-1])
    assert not state.requires_grad and state.shape==(108,width)
    # Fixed diagnostic readout exercises downstream gradient. This is NOT language QA.
    readout=t.randn(32,4);target=t.tensor([2])
    loss=t.nn.functional.cross_entropy((features.mean(0)@readout)[None],target)
    loss.backward();norm=float(t.sqrt(sum((p.grad**2).sum() for p in scorer.parameters())))
    assert norm>1e-8 and all(t.isfinite(p.grad).all() for p in scorer.parameters())
    assert all(p.grad is None for p in model.parameters())
    opt=t.optim.SGD(scorer.parameters(),lr=.01);opt.step()
    assert any(not t.equal(v,before[k]) for k,v in scorer.state_dict().items())
    assert all(t.equal(v,original[k]) for k,v in visual.state_dict().items())
    with t.no_grad():
        a=encode(visual,pixels,grid,scorer);b=encode(visual,pixels,grid,scorer)
        assert all(t.equal(u,v) for u,v in zip(a,b))
        # Only prior pair descriptor is accepted; no accumulated graph/history.
        c=encode(visual,pixels,grid,scorer,previous=state)
        assert c[3].numel()==108*width and not c[3].requires_grad
    # Finite difference verifies the log-bias gradient rather than only checking grad existence.
    z=t.randn(108,requires_grad=True);probe=t.randn_like(x)
    value=(biased_block(visual.blocks[1],x,pos,z)*probe).sum();g=t.autograd.grad(value,z)[0]
    j=int(g.abs().argmax());eps=.005
    with t.no_grad():
        plus=z.detach().clone();minus=plus.clone();plus[j]+=eps;minus[j]-=eps
        fd=float(((biased_block(visual.blocks[1],x,pos,plus)-biased_block(visual.blocks[1],x,pos,minus))*probe).sum()/(2*eps))
    assert abs(fd-float(g[j]))<max(2e-3,abs(fd)*.03),(fd,float(g[j]))
    result=dict(status='passed',constant_bias_max_errors=errors,scorer_grad_norm=norm,
        finite_difference=fd,autograd=float(g[j]),trainable_parameters=sum(p.numel() for p in scorer.parameters()),
        frozen_backbone_unchanged=True,hard_inference_repeat_exact=True,whole_groups_and_positions_preserved=True,
        score_after_layer=1,bias_layer=2,prune_after_layer=2,pretrained_loads=0,GPU=0,
        actual_language_QA_gradient_checked=False,capability_or_speed_claim=False,seconds=time.monotonic()-start)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
