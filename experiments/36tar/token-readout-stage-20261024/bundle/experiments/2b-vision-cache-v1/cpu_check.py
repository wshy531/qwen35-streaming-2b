"""Actual tiny Qwen visual layers; numerical/mechanism checks, no real weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/2b_vision_cache_prepare_20261024_01'
OUT.mkdir(parents=True, exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--receipt',default='CPU_mechanism.json')
args=parser.parse_args()
assert Path(args.receipt).name == args.receipt and args.receipt.endswith('.json')
assert not (OUT/args.receipt).exists()
REPO = ROOT/'repo' if (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'
ACTIVE = REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
sys.path.insert(0, str(ACTIVE))
import psutil
import torch
from qwen35_local.configuration_qwen3_5 import Qwen3_5VisionConfig
from qwen35_local.modeling_qwen3_5 import Qwen3_5VisionModel
from cacher import PairVisionCache

torch.set_num_threads(1)
torch.manual_seed(917)
began = time.perf_counter()
config = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
    patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
    num_position_embeddings=64, hidden_act='gelu_pytorch_tanh')
config._attn_implementation = 'sdpa'
vision = Qwen3_5VisionModel(config).eval()
original = vision.forward.__func__
weights = {name: value.clone() for name, value in vision.state_dict().items()}
grid = torch.tensor([[1, 12, 16]])
x = torch.randn(192, 3*2*16*16)
y = x + torch.randn_like(x)*0.05
checks = []


def close(a, b, label):
    error = float((a-b).abs().max())
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-5), (label, error)
    checks.append(dict(name=label, max_absolute_error=error))


with torch.inference_mode():
    native_x = vision(x, grid_thw=grid).pooler_output.clone()
    native_y = vision(y, grid_thw=grid).pooler_output.clone()
    with PairVisionCache(vision, update_ratio=1.0) as cache:
        close(vision(x, grid_thw=grid, return_dict=True).pooler_output, native_x, 'full_refresh_matches_native')
        close(vision(y, grid_thw=grid).pooler_output, native_y, 'all_positions_update_matches_native')
        assert not cache.last['full'] and cache.last['updated_tokens'] == 192
    assert vision.forward.__func__ is original
    with PairVisionCache(vision) as cache:
        close(vision(x, grid_thw=grid).pooler_output, native_x, 'cold_reference_matches_native')
        reference_copy = {i: {k: v.clone() for k, v in r.items()} for i, r in cache.references.items()}
        reference_pointers = {i: {k: v.data_ptr() for k, v in r.items()} for i, r in cache.references.items()}
        mlp_rows = []
        handles = [b.mlp.register_forward_pre_hook(lambda _m, args: mlp_rows.append(args[0].shape[0])) for b in vision.blocks]
        try:
            close(vision(x, grid_thw=grid).pooler_output, native_x, 'repeated_pair_partial_update')
        finally:
            for handle in handles:
                handle.remove()
        assert mlp_rows == [48, 48], mlp_rows
        assert cache.last['projected_tokens'] == 48 and cache.last['updated_tokens'] == 48
        assert cache.reference_bytes() == 4*2*192*32*4
        assert all(torch.equal(v, reference_copy[i][k]) for i, r in cache.references.items() for k, v in r.items())
        assert cache.full_pairs == cache.selective_pairs == 1
        before = cache.fingerprint()
        with cache.bypass():
            close(vision(y, grid_thw=grid, return_dict=True).pooler_output, native_y, 'bypass_is_native')
        assert cache.fingerprint() == before
        assert all(torch.equal(v, reference_copy[i][k]) for i, r in cache.references.items() for k, v in r.items())
        checks.append(dict(name='actual_MLP_rows_and_frozen_reference', rows=mlp_rows,
                           reference_bytes=cache.reference_bytes(), output_tokens=48))
        close(vision(y, grid_thw=grid).pooler_output, native_y, 'periodic_reference_refresh')
        assert cache.last['full'] and cache.last['reason'] == 'periodic'
        assert all(v.data_ptr() == reference_pointers[i][k] for i, r in cache.references.items() for k, v in r.items())
        # This is the approximate branch: record feature error without calling it accuracy.
        partial = vision(x, grid_thw=grid).pooler_output
        checks.append(dict(name='changed_pair_approximation',
            feature_max_absolute_error=float((partial-native_x).abs().max()),
            is_capability_test=False, updated_tokens=cache.last['updated_tokens']))
        cache.reset()
        assert cache.reference_bytes() == 0
        close(vision(y, grid_thw=grid).pooler_output, native_y, 'new_source_reset_matches_native')
        # Same token count but different spatial geometry must invalidate references.
        changed_grid = torch.tensor([[1, 16, 12]])
        output = vision(x, grid_thw=changed_grid).pooler_output
        assert cache.last['full'] and cache.last['reason'] == 'grid_or_tensor_change'
        checks.append(dict(name='same_size_grid_change_forces_refresh', shape=list(output.shape)))
        try:
            vision(x, grid_thw=torch.tensor([[2, 12, 16]]))
        except ValueError:
            pass
        else:
            raise AssertionError('packed temporal input was accepted')
        assert cache.reference_bytes() == 0 and cache.pair_index == 0
        checks.append(dict(name='invalid_input_clears_reference', passed=True))
    assert not cache.active and cache.reference_bytes() == 0
    close(vision(y, grid_thw=grid).pooler_output, native_y, 'unregister_restores_native')
    with PairVisionCache(vision) as outer:
        try:
            with PairVisionCache(vision):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError('two cache owners allowed')
        assert outer.active
    assert all(torch.equal(value, weights[name]) for name, value in vision.state_dict().items())
    checks.append(dict(name='weights_unchanged_and_duplicate_owner_rejected', passed=True))

# Rejection is checked outside inference mode and must leave no stale reference.
with PairVisionCache(vision) as cache:
    try:
        vision(x, grid_thw=grid)
    except ValueError:
        pass
    else:
        raise AssertionError('gradients were accepted')
    assert cache.reference_bytes() == 0
checks.append(dict(name='autograd_rejected', passed=True))
peak = getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)
assert peak < 1536*2**20 and time.perf_counter()-began < 600
actual = json.loads((ROOT/'models/Qwen3.5-2B/config.json').read_text())['vision_config']
budget = 4*actual['depth']*192*actual['hidden_size']*4
assert budget == 72*2**20
report = dict(status='passed', checks=checks, pretrained_loads=0, GPU_jobs=0,
    random_tiny_vision_only=True, actual_2B_reference_bytes_FP32=budget,
    peak_process_RSS_bytes=peak, elapsed_seconds=time.perf_counter()-began,
    torch=torch.__version__, source_SHA256=hashlib.sha256((Path(__file__).parent/'cacher.py').read_bytes()).hexdigest())
with (OUT/args.receipt).open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, ensure_ascii=False)
print(json.dumps(report, ensure_ascii=False, indent=2))
