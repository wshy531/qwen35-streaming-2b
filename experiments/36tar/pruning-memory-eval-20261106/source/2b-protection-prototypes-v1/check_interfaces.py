import sys,os,json,hashlib
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-prune-timing-audit-v1'))
from audit_common import runtime,make_builder
for name in ['2b-anchor-merge-v1','2b-isolated-recent-v1']:
    sys.path.insert(0,str(R/'experiments'/name))
from encoder import encode
from content_prune import encode as old_encode,CoverageScorer
from protected_readout import answer
from quality_banks import selected_node
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
processor=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval();model.requires_grad_(False)
builder=make_builder(env,processor,(288,384))
import numpy as np
rng=np.random.default_rng(20261103);scorers=[CoverageScorer() for _ in range(3)];nodes=[]
weight_hash=lambda:hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in model.parameters())).hexdigest()
before=weight_hash()
with t.inference_mode(),env['fp32_reference']():
    for i in range(3):
        frames=[SimpleNamespace(frame_id=2*i+k,timestamp=float(2*i+k),rgb=rng.integers(0,256,(288,384,3),np.uint8)) for k in (0,1)]
        inputs,_=builder.pair(*frames,0,0)
        x,ids,trace=encode(model.model.visual,inputs.pixel_values_videos,inputs.video_grid_thw,scorer=scorers[0],enabled=False)
        ref,ri,_=old_encode(model.model.visual,inputs.pixel_values_videos,inputs.video_grid_thw,mode='mid_content',scorer=scorers[1])
        merged,mi,mt=encode(model.model.visual,inputs.pixel_values_videos,inputs.video_grid_thw,scorer=scorers[2])
        assert t.equal(x,ref) and t.equal(ids,ri) and t.equal(ids,mi)
        assert t.equal(scorers[0].previous,scorers[2].previous) and sum(mt['merge_mass'])==108
        assert mt['layer_tokens']==[432,432,216,216] and t.isfinite(merged).all()
        assert not model.model.visual.blocks[2]._forward_pre_hooks
        nodes.append(selected_node(ref,ri,i,float(2*i),float(2*i+1)))
    question='What is visible now? A. A person. B. A cup.'
    joint=answer(env,model,builder,nodes[:1],nodes[1:],question,isolated=False,limit=2)
    isolated=answer(env,model,builder,nodes[:1],nodes[1:],question,isolated=True,limit=2)
    for key in ('input_ids','position_ids'):assert t.equal(joint[key],isolated[key])
    for row in (joint,isolated):assert row['branch_released'] and row['read_state_unchanged'] and t.isfinite(row['first_logits']).all()
    a=answer(env,model,builder,[],nodes,question,isolated=False,limit=2)
    b=answer(env,model,builder,[],nodes,question,isolated=True,limit=2)
    assert t.equal(a['first_logits'],b['first_logits']) and a['generated_ids']==b['generated_ids']
assert weight_hash()==before
out=R/'runs/protection_interfaces_20261103_01';out.mkdir(exist_ok=False)
(out/'CPU_local.json').write_text(json.dumps(dict(status='passed',RGB_pairs=3,visual_calls=9,old_encoder_exact=True,
    identical_selected_IDs=True,same_scorer_state=True,hook_removed=True,weights_unchanged=True,
    tiny_visual_QA=4,joint_and_isolated_positions_identical=True,empty_history_logits_exact=True,
    joint_cache_bytes=joint['temporary_cache_bytes_at_end'],isolated_cache_bytes=isolated['temporary_cache_bytes_at_end'],
    GPU=0,pretrained_loads=0,capability_claim=False),indent=2))
print('Both RGB/visual QA interfaces passed; no pretrained capability claim')
