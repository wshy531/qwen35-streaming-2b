"""Paired whole-graph/recomputed preflight; no optimizer or generated answers."""
import argparse
import gc
import json
import os
import time
from pathlib import Path
import torch
from profile_delayed_qa import (ROOT, PREP, WEIGHTS, CHECKPOINT, runtime, make_builder,
                                GroupScorer, read, sha, decode, run_case, question_text)
from split_backward import capture_pairs, qa_leaf_gradients, visual_backward, cache_bytes

NEW_PREP = ROOT / 'runs/delayed_recompute_prepare_20261104_01'


def clear_and_reset(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-tiny', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to((ROOT / 'runs').resolve())
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_tiny:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1
    for path, digest in read(NEW_PREP / 'execution_lock.json').items():
        assert sha(ROOT / path) == digest, path
    cases = read(PREP / 'cases.json')
    assert len(cases) == 2 and sum(len(c['audit']['samples'])//2 for c in cases) == 38
    args.out.mkdir()
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(WEIGHTS, local_files_only=True)
    builder = make_builder(env, processor)
    if args.cpu_tiny:
        config = env['tiny_config'](processor)
        config['vision_config']['depth'] = 24
        model = env['Qwen3_5ForConditionalGeneration'](
            env['Qwen3_5Config'](**config, attn_implementation='sdpa')).float()
    else:
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            WEIGHTS, local_files_only=True, dtype=torch.float32, device_map='cuda',
            attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False)
    versions = {n:p._version for n,p in model.named_parameters()}
    device = model.get_input_embeddings().weight.device
    scorer = GroupScorer(model.config.vision_config.hidden_size).to(device)
    if not args.cpu_tiny:
        scorer.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    with (args.out / 'cases.jsonl').open('x', encoding='utf-8') as handle:
        for case in cases:
            source = case['source']
            assert source['role'] == 'train'
            path = ROOT / 'datasets/learned-prune-nextqa-v1' / source['mirror_path']
            assert sha(path) == source['mirror_lfs_sha256']
            pairs = decode(env, path, case['audit'], builder)
            clear_and_reset(device)
            with env['fp32_reference']():
                reference = run_case(model, builder, scorer, pairs, case)
            gradients = {n:p.grad.detach().cpu().clone() for n,p in scorer.named_parameters()}
            scorer.zero_grad(set_to_none=True)
            # The old BatchFeature.to() mutates its input; release those GPU pixels.
            for _, _, batch in pairs:
                batch.to('cpu')
            inputs = [(begin,end,batch.pixel_values_videos,batch.video_grid_thw)
                      for begin,end,batch in pairs]
            clear_and_reset(device)
            started = time.monotonic()
            with env['fp32_reference']():
                captured = capture_pairs(model.model.visual, scorer, inputs)
                observed = qa_leaf_gradients(model, builder, captured, question_text(case['question']),
                                             case['question']['answer_index'])
                score_gradients = visual_backward(model.model.visual, scorer, captured,
                                                  observed['gradients'], inputs.__getitem__)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.monotonic()-started
            deltas = {}
            for n,p in scorer.named_parameters():
                actual = p.grad.detach().cpu()
                expected = gradients[n]
                delta = actual-expected
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=5e-4)
                rel_l2 = float(delta.norm()/expected.norm().clamp_min(1e-12))
                assert rel_l2 <= 5e-4, (n,rel_l2)
                deltas[n] = dict(max_abs=float(delta.abs().max()), relative_l2=rel_l2)
            assert abs(observed['loss']-reference['ce']) <= 2e-4
            assert observed['metadata']['visual_tokens'] == reference['visual_tokens']
            old_gradient = sum(float(score_gradients[i].abs().sum()) for i in case['evidence_pairs'])
            assert old_gradient > 0
            assert all(p.grad is None and p._version == versions[n] for n,p in model.named_parameters())
            result = dict(id=case['question']['id'], whole=reference,
                recompute=dict(ce=observed['loss'],seconds=seconds,old_score_grad_l1=old_gradient,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else None,
                    cache_tensor_bytes=cache_bytes(captured),
                    leaf_gradient_bytes=sum(g.numel()*g.element_size() for g in observed['gradients'].values())),
                param_gradient_deltas=deltas,gradient_atol=2e-6,gradient_rtol=5e-4,
                relative_l2_limit=5e-4,updates=0)
            handle.write(json.dumps(result)+'\n')
            handle.flush()
            print(json.dumps(result), flush=True)
            del captured, observed, score_gradients, gradients, inputs, pairs
    (args.out / 'completed.json').write_text(json.dumps(dict(status='passed',cpu_tiny=args.cpu_tiny,
        cases=2,visual_forwards=114,language_forwards=6,qa_backwards=4,
        recomputed_visual_backwards=38,updates=0,generated_tokens=0,
        limits='fixed schedule and same surrogate; cache and CPU pixels grow with clip length'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
