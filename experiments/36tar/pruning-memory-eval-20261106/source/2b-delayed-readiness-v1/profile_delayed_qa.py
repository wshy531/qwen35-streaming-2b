"""Bounded delayed-QA preflight. No optimizer, generation, or downloads."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from replay import ROOT, Replay, banks
from delayed_qa import last_logits, training_context
from resolution_runtime import context, make_builder

for name in ('2b-hard-mask-prune-v1', '2b-midprune-memory-v1',
             '2b-learned-prune-main-v1'):
    sys.path.insert(0, str(ROOT / 'experiments' / name))
from adapter import sparse_pair
from hard_prune import GroupScorer, encode
from objective import loss_parts, option_ids
from protocol import decode, question_text, read, runtime, sha
import torch

PREP = ROOT / 'runs/delayed_profile_prepare_20261104_01'
WEIGHTS = ROOT / 'models/Qwen3.5-2B'
CHECKPOINT = ROOT / 'runs/hard-mask-main-220062/mask_ce_step1237.pt'


def run_case(model, builder, scorer, pairs, case):
    device = model.get_input_embeddings().weight.device
    bank = banks.Bank('LS', 54, budget=864, recent=432, closed=324)
    replay = Replay()
    leaves, scores = {}, {}
    state = None
    started = time.monotonic()
    scorer.zero_grad(set_to_none=True)
    with replay.capture():
        for index, (begin, end, batch) in enumerate(pairs):
            batch = batch.to(device)
            features, kept, scores[index], state = encode(
                model.model.visual, batch.pixel_values_videos, batch.video_grid_thw,
                scorer, previous=state, mode='mask')
            scores[index].retain_grad()
            leaves[index] = features
            bank.append(sparse_pair(features, kept, index, begin, end),
                        starts_event=index % 3 == 0)
    nodes = bank.export()
    evidence = case['evidence_pairs']
    assert max(evidence) < len(pairs) - 8
    assert all(any(n.first <= index <= n.last for n in nodes) for index in evidence)
    assert any(n.first < n.last and n.first <= evidence[0] <= n.last for n in nodes)
    prompt = question_text(case['question'])
    emb, pos, ids, meta = training_context(model, builder, nodes, prompt, replay, leaves)
    with torch.no_grad():
        ref_emb, ref_pos, ref_ids, _ = context(model, builder, nodes, prompt)
        reference = last_logits(model, ref_emb, ref_pos)
    assert torch.equal(pos, ref_pos) and torch.equal(ids, ref_ids)
    torch.testing.assert_close(emb, ref_emb, atol=2e-5, rtol=5e-5)
    logits = last_logits(model, emb, pos)
    torch.testing.assert_close(logits, reference, atol=2e-4, rtol=5e-5)
    loss, _, _, _ = loss_parts(logits, option_ids(builder, device), case['question']['answer_index'])
    loss.backward()
    grads = [scores[i].grad for i in evidence]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    old_gradient = sum(float(g.abs().sum()) for g in grads)
    assert old_gradient > 0 and all(p.grad is None for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scorer.parameters())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return dict(id=case['question']['id'], pairs=len(pairs), merges=len(replay.steps),
                old_evidence_score_grad_l1=old_gradient, tokens=ids.numel(),
                visual_tokens=meta['visual_tokens'], ce=float(loss.detach()),
                max_logit_delta=float((logits.detach()-reference).abs().max()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else None,
                seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-tiny', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    assert args.out.resolve().is_relative_to((ROOT / 'runs').resolve())
    if not args.cpu_tiny:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    for path, digest in read(PREP / 'execution_lock.json').items():
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
    versions = {n: p._version for n, p in model.named_parameters()}
    scorer = GroupScorer(model.config.vision_config.hidden_size).to(model.get_input_embeddings().weight.device)
    if not args.cpu_tiny:
        scorer.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    with (args.out / 'cases.jsonl').open('x', encoding='utf-8') as log:
        for case in cases:
            source = case['source']
            assert source['role'] == 'train'
            path = ROOT / 'datasets/learned-prune-nextqa-v1' / source['mirror_path']
            assert sha(path) == source['mirror_lfs_sha256']
            pairs = decode(env, path, case['audit'], builder)
            assert 17 <= len(pairs) <= 24
            if not args.cpu_tiny:
                torch.cuda.reset_peak_memory_stats()
            with env['fp32_reference']():
                result = run_case(model, builder, scorer, pairs, case)
            assert all(p._version == versions[n] for n, p in model.named_parameters())
            print(json.dumps(result), flush=True)
            log.write(json.dumps(result)+'\n')
            log.flush()
    (args.out / 'completed.json').write_text(json.dumps(dict(
        status='passed', cpu_tiny=args.cpu_tiny, pairs=38, language_forwards=4,
        backwards=2, updates=0, generated_tokens=0), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
