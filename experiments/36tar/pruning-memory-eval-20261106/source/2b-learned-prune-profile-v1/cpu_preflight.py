"""Same context as inference, with a real autograd path through two visual pairs."""
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for name in ['2b-sember-v1','2b-resolution-cost-v1','2b-learned-prune-mechanism-v1']:
    sys.path.insert(0,str(ROOT/'experiments'/name))
from sember_common import runtime
from resolution_runtime import make_builder,context as reference_context
from resolution_nodes import Node
from bias_prune import GroupScorer,encode
from training import context,task_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261101)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    scorer=GroupScorer(model.config.vision_config.hidden_size);builder=make_builder(env,processor)
    records=[];nodes=[];state=None
    for index in range(2):
        features,ids,_,state=encode(model.model.visual,t.randn(432,1536),t.tensor([[1,18,24]]),scorer,previous=state)
        assert not state.requires_grad
        begin,end=float(index),index+.5
        records.append((index,begin,end,features,ids))
        coords=t.stack((t.full_like(ids,index),ids//12,ids%12),1)
        nodes.append(Node(features.detach().clone(),t.ones((54,1)),coords,index,index,begin,end,(begin+end)/2))
    q='What is happening now?\nA. moving\nB. sitting\nAnswer with one option letter.'
    emb,pos,ids=context(model,builder,records,q)
    ref,rpos,rids,_=reference_context(model,builder,nodes,q,visual_limit=108)
    assert t.equal(emb,ref) and t.equal(pos,rpos) and t.equal(ids,rids)
    with env['fp32_reference']():loss,tokens=task_loss(model,builder,records,q,'A');loss.backward()
    norm=float(t.sqrt(sum((p.grad**2).sum() for p in scorer.parameters() if p.grad is not None)))
    assert norm>1e-12 and all(p.grad is None for p in model.parameters())
    result=dict(status='passed',two_pair_context_exact_to_existing_inference=True,input_tokens=tokens,
        scorer_gradient_norm=norm,previous_state_detached=True,frozen_base_no_grad=True,
        synthetic_pixels=True,pretrained_loads=0,GPU=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
