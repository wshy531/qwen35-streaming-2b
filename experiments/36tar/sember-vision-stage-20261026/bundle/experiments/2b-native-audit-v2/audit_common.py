"""Independent original-backend audit; no project model imports here."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/native_audit_prepare_20261025_02'
MODEL = ROOT / 'models/Qwen3.5-2B'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=json_default)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def official_identity(model=None):
    import inspect
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as module
    path = Path(module.__file__).resolve()
    package = Path(transformers.__file__).resolve().parent
    assert path.is_relative_to(package)
    assert not any(name == 'qwen35_local' or name.startswith(('qwen35_local.', 'streaming.')) for name in sys.modules)
    if model is not None:
        assert inspect.getmodule(type(model)) is module
        assert model.forward.__func__ is type(model).forward
    return dict(transformers_version=transformers.__version__, model_source=str(path),
                model_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                project_model_or_streaming_imported=False)


def tiny_config(processor):
    # Same fixed small dimensions as the previous numeric check, not real weights.
    return dict(text_config=dict(vocab_size=248320, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=['linear_attention', 'full_attention'], linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        rope_parameters=dict(rope_type='default', rope_theta=10000., partial_rotary_factor=1., mrope_section=[2,3,3])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
            patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
            out_hidden_size=32, num_position_embeddings=16),
        image_token_id=processor.image_token_id, video_token_id=processor.video_token_id,
        vision_start_token_id=processor.tokenizer.convert_tokens_to_ids(processor.vision_start_token),
        vision_end_token_id=processor.tokenizer.convert_tokens_to_ids(processor.vision_end_token), tie_word_embeddings=False)


def template(processor, question):
    return processor.apply_chat_template([dict(role='user', content=[dict(type='video'),
        dict(type='text', text=question)])], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def paired_inputs(processor, frames, timestamps, question):
    """Reproduce the current input protocol, without importing its execution code."""
    import numpy as np
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    assert len(frames) == len(timestamps) and len(frames) % 2 == 0 and len(frames) >= 2
    marker = processor.vision_start_token + processor.video_token + processor.vision_end_token
    prefix, suffix = template(processor, question).split(marker)
    blocks, pixels, grids = [], [], []
    for i in range(0, len(frames), 2):
        result = processor.video_processor(videos=[np.stack(frames[i:i+2])], do_resize=False,
            do_sample_frames=False, return_metadata=False, return_tensors='pt')
        grid = result.video_grid_thw
        assert grid.shape == (1,3) and int(grid[0,0]) == 1
        count = int(grid[0,1:].prod()) // processor.video_processor.merge_size**2
        blocks.append(f'<{(timestamps[i]+timestamps[i+1])/2:.1f} seconds>' +
            processor.vision_start_token + processor.video_token * count + processor.vision_end_token)
        pixels.append(result.pixel_values_videos)
        grids.append(grid)
    ids = processor.tokenizer(prefix + ''.join(blocks) + suffix, add_special_tokens=False, return_tensors='pt').input_ids
    return BatchFeature(dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        mm_token_type_ids=(ids == processor.video_token_id).long()*2,
        pixel_values_videos=torch.cat(pixels), video_grid_thw=torch.cat(grids)))


def json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")
