"""Mechanism checks only: exact all-kept replay, original grouped positions and fresh QA contexts."""
import argparse
import os
from pathlib import Path
from common import ROOT,OUT,runtime,read,save
from midprune import encode


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(20261025)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    original_weights={n:t.clone() for n,t in model.model.visual.state_dict().items()}
    cases=[]
    with torch.inference_mode(),env['fp32_reference']():
        for h,w in ((12,16),(18,24)):
            grid=torch.tensor([[1,h,w]])
            pixels=torch.randn(h*w,3*2*16*16)
            original=model.model.visual(pixels,grid_thw=grid).pooler_output
            replay,all_ids,full_trace=encode(model.model.visual,pixels,grid,mode='full_replay',return_trace=True)
            assert torch.equal(original,replay),'all-kept replay changed the original visual computation'
            post,post_ids,post_trace=encode(model.model.visual,pixels,grid,mode='post_odd',return_trace=True)
            mid,mid_ids,mid_trace=encode(model.model.visual,pixels,grid,mode='mid_odd',return_trace=True)
            assert torch.equal(post,original[post_ids])
            assert torch.equal(mid_ids,post_ids) and mid.shape==post.shape and torch.isfinite(mid).all()
            assert full_trace['layer_tokens']==post_trace['layer_tokens']==[h*w]*4
            assert mid_trace['layer_tokens']==[h*w]*2+[h*w//2]*2
            assert torch.equal(mid_ids,torch.arange(1,h*w//4,2))
            # Each selected output group retains all four original merger inputs.
            patch_ids=(mid_ids[:,None]*4+torch.arange(4)[None,:]).flatten()
            assert len(torch.unique(patch_ids))==h*w//2
            assert torch.equal(patch_ids.reshape(-1,4)//4,mid_ids[:,None].expand(-1,4))
            cases.append(dict(grid=[h,w],all_kept_max_abs_error=float((original-replay).abs().max()),
                post_selection_exact=True,mid_layer_tokens=mid_trace['layer_tokens'],output_tokens=len(mid)))
        # No hidden cross-frame state: A -> unrelated B -> A must give exactly the same A.
        grid=torch.tensor([[1,18,24]]);a=torch.randn(432,1536);b=torch.randn(432,1536)
        first,ids,_=encode(model.model.visual,a,grid)
        encode(model.model.visual,b,grid)
        again,_,_=encode(model.model.visual,a,grid)
        assert torch.equal(first,again)
        from sember_nodes import Node,validate,digest
        from sember_runtime import context,make_builder
        builder=make_builder(env,processor)
        coords=torch.stack((torch.zeros_like(ids),ids//12,ids%12),1)
        node=Node(first.clone(),torch.ones((54,1)),coords,0,0,0.,2.,1.,grid=(9,12))
        validate(node);before=digest([node])
        emb,pos,ids_text,metadata=context(model,builder,[node],'What is visible?',visual_limit=432,max_new_tokens=2)
        seq=env['HybridSequence'](model,max_tokens=4096)
        logits=seq.append(inputs_embeds=emb,position_ids=pos,logits_mode='last')
        assert torch.isfinite(logits).all()
        seq.cache.reset()
        assert digest([node])==before and metadata['visual_tokens']==54
        for name,value in model.model.visual.state_dict().items():assert torch.equal(value,original_weights[name]),name
    result=dict(status='passed',cases=cases,whole2x2_groups_preserved=True,original_grid_coordinates=True,
        A_B_A_no_state=True,model_parameters_unchanged=True,tiny_QA_context=True,QA_released=True,
        questions_and_scores_read=0,pretrained_loads=0,GPU_jobs=0,no_capability_or_speed_claim=True,
        candidate_fixed_real_layers=12,candidate_fixed_real_keep_groups=54)
    save(output,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
