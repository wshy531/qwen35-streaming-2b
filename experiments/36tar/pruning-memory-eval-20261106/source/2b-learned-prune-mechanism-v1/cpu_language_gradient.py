"""Real tiny hybrid-language loss through selected visual features, synthetic token target."""
import os,sys,json,argparse
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import runtime
from bias_prune import GroupScorer,encode

def main(out):
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261102)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    model.requires_grad_(False);visual=model.model.visual;scorer=GroupScorer(visual.config.hidden_size)
    pixels=t.randn(432,1536);grid=t.tensor([[1,18,24]])
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    ids=t.tensor([[10,11,12]])
    with env['fp32_reference']():
        features,selected,_,_=encode(visual,pixels,grid,scorer)
        features.retain_grad()
        embeds=t.cat((features[None],model.get_input_embeddings()(ids)),dim=1)
        n=embeds.shape[1];positions=t.arange(n)[None,None,:].expand(4,1,-1).clone()
        # Original spatial grid positions for the visual prefix; text after max visual coordinate.
        positions[1,0,:54]=0;positions[2,0,:54]=selected//12;positions[3,0,:54]=selected%12
        positions[1:,0,54:]=t.arange(12,15)[None,:]
        output=model.model.language_model(inputs_embeds=embeds,position_ids=positions,
            attention_mask=t.ones((1,n),dtype=t.long),use_cache=False)
        logits=model.lm_head(output.last_hidden_state[:,-1])
        loss=t.nn.functional.cross_entropy(logits,t.tensor([13]));loss.backward()
    norm=float(t.sqrt(sum((p.grad**2).sum() for p in scorer.parameters())))
    visual_grad=float(features.grad.norm())
    assert norm>1e-10 and visual_grad>1e-10
    assert all(p.grad is None for p in model.parameters())
    assert all(t.equal(v,before[k]) for k,v in model.state_dict().items())
    result=dict(status='passed',actual_hybrid_language_loss_gradient=True,scorer_grad_norm=norm,
        visual_features_grad_norm=visual_grad,frozen_all_model_parameters_unchanged=True,
        synthetic_token_target=True,trained_language_capability=False,real_pixels_or_pretrained_weights=False,
        selected_groups=54,GPU=0,pretrained_loads=0,base_updates=0)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
