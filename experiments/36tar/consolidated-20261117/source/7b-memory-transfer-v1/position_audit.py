"""CPU-only regression: exercise the installed official position implementation."""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

ROOT = Path(__file__).resolve().parents[2]

def main():
    # No model weights, tokenizer, patched function, or GPU is used.
    stub = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(
        spatial_merge_size=2, tokens_per_second=2),vision_start_token_id=90,
        image_token_id=93,video_token_id=91))
    if hasattr(Qwen2_5_VLModel,'get_vision_position_ids'):
        stub.get_vision_position_ids = lambda *a, **kw: Qwen2_5_VLModel.get_vision_position_ids(stub, *a, **kw)
    rows = []
    for t in (1, 3):
        for seconds in (0.5, 2.0):
            h, w, prefix, suffix = 2, 3, 5, 2
            length = prefix + t*h*w + suffix
            ids = torch.ones((1, length), dtype=torch.long)
            ids[:,prefix-1]=90
            ids[:,prefix:-suffix]=91
            ids[:,-suffix]=92
            types = torch.zeros_like(ids)
            types[:, prefix:-suffix] = 2
            actual, delta = Qwen2_5_VLModel.get_rope_index(
                stub, input_ids=ids, mm_token_type_ids=types,
                video_grid_thw=torch.tensor([[t,h*2,w*2]]),
                second_per_grid_ts=torch.tensor([seconds]))
            temporal = (torch.arange(t)*seconds*2).long().repeat_interleave(h*w)
            height = torch.arange(h).repeat_interleave(w).repeat(t)
            width = torch.arange(w).repeat(h*t)
            visual = torch.stack((temporal,height,width)) + prefix
            expected = torch.cat((torch.arange(prefix).expand(3,-1),visual,
                (torch.arange(suffix)+visual.max()+1).expand(3,-1)), dim=1).unsqueeze(1)
            rows.append(dict(temporal_patches=t, seconds_per_patch=seconds,
                equal=torch.equal(actual,expected),
                temporal_actual=actual[0,0,prefix:-suffix].tolist(),
                temporal_expected=visual[0].tolist(),
                spatial_equal=torch.equal(actual[1:,0,prefix:-suffix],visual[1:]),
                suffix_equal=torch.equal(actual[:,:,-suffix:],expected[:,:,-suffix:]),
                delta_equal=torch.equal(delta, (expected.max()+1-length).reshape(1,1))))
    source = Path(inspect.getfile(Qwen2_5_VLModel))
    result = dict(version=transformers.__version__, source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        all_passed=all(r['equal'] and r['delta_equal'] for r in rows),
        planned_fps1_passed=all(r['equal'] and r['delta_equal'] for r in rows if r['seconds_per_patch']==2), rows=rows,
        GPU=0, real_model_loaded=False,
        upstream_fix='https://github.com/huggingface/transformers/pull/45400')
    out = ROOT/'runs/7b_transfer_prepare_20261107_01'/f'position_audit_{transformers.__version__}.json'
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__ == '__main__': main()
