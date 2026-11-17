"""Exercise actual Qwen2.5 geometry/depth with reduced random widths on CPU."""
import hashlib
import json
from pathlib import Path
import torch
import transformers
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'

def make_tiny(real_tokens=False):
    cfg=json.loads((PREP/'config.json').read_text())
    assert hashlib.sha256((PREP/'config.json').read_bytes()).hexdigest()=='77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43'
    cfg['torch_dtype']='float32'
    cfg.update(hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,
        rope_scaling={'type':'mrope','mrope_section':[2,3,3]})
    if not real_tokens:
        cfg.update(vocab_size=256,bos_token_id=1,eos_token_id=2,vision_start_token_id=90,
            video_token_id=91,vision_end_token_id=92,image_token_id=93)
    cfg['vision_config'].update(hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=64)
    return Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**cfg)).eval().requires_grad_(False)

def main():
    assert transformers.__version__=='5.2.0'
    assert json.loads((PREP/'position_audit_5.2.0.json').read_text())['planned_fps1_passed']
    torch.set_num_threads(1); torch.manual_seed(7007)
    model=make_tiny()
    processor=Qwen2VLVideoProcessor(do_resize=False,do_sample_frames=False)
    frames=torch.randint(0,256,(8,3,280,392),dtype=torch.uint8)
    batch=processor(videos=[frames],return_tensors='pt')
    parts=[processor(videos=[frames[i:i+2]],return_tensors='pt') for i in range(0,8,2)]
    assert batch['video_grid_thw'].tolist()==[[4,20,28]]
    assert torch.equal(batch['pixel_values_videos'],torch.cat([b['pixel_values_videos'] for b in parts]))
    called=[]
    hooks=[b.register_forward_hook(lambda mod,args,out,i=i:called.append(i)) for i,b in enumerate(model.model.visual.blocks)]
    with torch.no_grad():
        full=model.get_video_features(**batch,return_dict=True).pooler_output[0]
        assert called==list(range(32))
        for h in hooks:h.remove()
        incremental=torch.cat([model.get_video_features(**b,return_dict=True).pooler_output[0] for b in parts])
        assert full.shape==(560,64)
        torch.testing.assert_close(full,incremental,atol=1e-6,rtol=1e-5)
        ids=torch.tensor([[1,15,90]+[91]*560+[92,17,18]])
        mask=torch.ones_like(ids)
        pos,_=model.model.get_rope_index(ids,video_grid_thw=batch['video_grid_thw'],second_per_grid_ts=torch.tensor([2.0]))
        native=model(input_ids=ids,attention_mask=mask,**batch,
            second_per_grid_ts=torch.tensor([2.0]),use_cache=False,logits_to_keep=1).logits
        embeddings=model.get_input_embeddings()(ids)
        embeddings[:,ids[0]==91]=incremental
        direct=model.model.language_model(inputs_embeds=embeddings,position_ids=pos,
            attention_mask=mask,use_cache=False).last_hidden_state
        projected=model.lm_head(direct[:,-1:])
        torch.testing.assert_close(native,projected,atol=1e-6,rtol=1e-5)
        cache=None
        for begin in range(0,ids.shape[1],512):
            end=min(begin+512,ids.shape[1])
            piece=model.model.language_model(inputs_embeds=embeddings[:,begin:end],
                position_ids=pos[:,:,begin:end],attention_mask=mask[:,:end],
                past_key_values=cache,use_cache=True)
            cache=piece.past_key_values
        chunked=model.lm_head(piece.last_hidden_state[:,-1:])
        torch.testing.assert_close(native,chunked,atol=1e-6,rtol=1e-5)
        # Exercise official generation too, not just a first-logit assertion.
        model.model.rope_deltas=None
        generated=model.generate(input_ids=ids,attention_mask=mask,**batch,
            second_per_grid_ts=torch.tensor([2.0]),max_new_tokens=4,min_new_tokens=4,
            do_sample=False,pad_token_id=0,eos_token_id=2)
        assert generated.shape[1]==ids.shape[1]+4
    result=dict(status='CPU mechanism passed',version=transformers.__version__,
        actual_weights=False,synthetic_pixels=True,canvas_hw=[280,392],grid=[4,20,28],
        full_vision_depth=32,vision_calls=6,visual_tokens_per_pair=140,
        pair_pixel_exact=True,visual_max_abs=float((full-incremental).abs().max()),
        native_embeddings_logit_max_abs=float((native-projected).abs().max()),
        chunk512_logit_max_abs=float((native-chunked).abs().max()),
        native_generated_ids=generated[0,-4:].tolist(),GPU=0)
    (PREP/'native_cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
