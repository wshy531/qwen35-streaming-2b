"""Compare four frozen pruning paths with one causal S-EMBER LS protocol."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ('2b-sember-main-v1', '2b-sember-v1', '2b-hard-mask-prune-v1',
             '2b-content-midprune-v1'):
    sys.path.insert(0, str(ROOT / 'experiments' / name))
from main_common import DATA, OUT, read, runtime, save, sha, verify
from evaluate_main import group_queries
from content_prune import encode as fixed_encode
from hard_prune import GroupScorer, prefix, selected_ids, tail
from sember_banks import Bank
from sember_nodes import Node, digest
from sember_protocol import prompt
from sember_runtime import context, make_builder
from sember_session import emit_decode
from sember_source import CausalSource, PairDriver
import torch

PREP = ROOT / 'runs/sember_pruned_ls_prepare_20261104_01'
CHECKPOINTS = {
    'guarded': 'runs/repair-guard-220499/final.pt',
    'old_CE': 'runs/hard-mask-main-220062/mask_ce_step1237.pt',
    'initial': 'runs/training-order-main-220483/initial.pt',
}
ARMS = (*CHECKPOINTS, 'fixed12')
PREFIX_FIELDS = ('pairs', 'real_frames', 'last_committed_pts',
                 'submitted_prefix_sha256', 'true_EOF_tail_padded')


class Observer:
    def __init__(self, model, builder, scorers):
        self.model = model
        self.builder = builder
        self.scorers = scorers
        self.banks = {arm: Bank('LS', 54, budget=864, recent=432, closed=324) for arm in ARMS}
        self.previous = dict.fromkeys(scorers)
        self.pairs = 0

    def append(self, first, second, *, padding=False):
        device = self.model.get_input_embeddings().weight.device
        inputs, _ = self.builder.pair(first, second, 0, 0, synthetic_padding=padding)
        pixels = inputs.pixel_values_videos.to(device)
        grid = inputs.video_grid_thw.to(device)
        visual = self.model.model.visual
        observed, layer12, positions = prefix(visual, pixels, grid)
        for arm, scorer in self.scorers.items():
            scores, self.previous[arm] = scorer(observed, self.previous[arm])
            ids = selected_ids(scores, 54)
            features = tail(visual, layer12, positions, scores, ids, 'mask')
            self.store(arm, features, ids, first.timestamp, second.timestamp)
        features, ids, _ = fixed_encode(visual, pixels, grid, mode='mid_fixed')
        assert ids.tolist() == list(range(1, 108, 2))
        self.store('fixed12', features, ids, first.timestamp, second.timestamp)
        self.pairs += 1

    def store(self, arm, features, ids, begin, end):
        ids = ids.detach().cpu()
        coords = torch.stack((torch.full_like(ids, self.pairs), ids // 12, ids % 12), dim=1)
        node = Node(features.detach().float().cpu().clone(), torch.ones(54, 1),
                    coords, self.pairs, self.pairs, begin, end, (begin + end) / 2)
        self.banks[arm].append(node, starts_event=self.pairs % 3 == 0)


def answer(env, model, builder, bank, question, limit=16):
    nodes = bank.export()
    fingerprint = digest(bank.nodes())
    saved = model.model.rope_deltas
    branch = env['HybridSequence'](model, max_tokens=4096, session_id='S-EMBER-pruned-LS')
    try:
        embeddings, positions, ids, meta = context(
            model, builder, nodes, question, visual_limit=864, max_new_tokens=limit)
        for begin in range(0, embeddings.shape[1], 512):
            end = min(begin + 512, embeddings.shape[1])
            logits = branch.append(inputs_embeds=embeddings[:, begin:end],
                                   position_ids=positions[:, :, begin:end],
                                   logits_mode='last' if end == embeddings.shape[1] else 'none')
        eos = model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        stops.add(builder.processor.tokenizer.eos_token_id)
        result = emit_decode(branch, logits, builder.processor, limit, stops, lambda item: None)
    finally:
        branch.cache.reset()
        model.model.rope_deltas = saved
    assert digest(bank.nodes()) == fingerprint
    return {**result, **meta, 'question_suffix_tokens': builder.question_ids(question).shape[1]}


def inputs(builder):
    groups = group_queries(read(OUT / 'inputs/requests.json'))
    expected = read(OUT / 'inputs/expected_prefixes.json')
    videos = {row['video_id']: row for row in read(OUT / 'inputs/videos.json')}
    assert len(groups) == len(videos) == 128 and sum(len(qs) for _, qs in groups) == 478
    assert sum(expected[qs[-1]['question_id']]['pairs'] for _, qs in groups) == 14678
    for vid, questions in groups:
        assert vid in videos
        for q in questions:
            assert expected[q['question_id']]['last_committed_pts'] <= q['question_time']
            assert builder.question_ids(prompt(q)).shape[1] <= 512
    return groups, expected, videos


def main(args):
    assert args.out.resolve().is_relative_to((ROOT / 'runs').resolve()) and not args.out.exists()
    if args.cpu_check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1
    for rel, expected in read(PREP / 'execution_lock.json').items():
        assert sha(ROOT / rel) == expected, rel
    verify()
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    env = runtime()
    weights = ROOT / 'models/Qwen3.5-2B'
    builder = make_builder(env, env['AutoProcessor'].from_pretrained(weights, local_files_only=True))
    groups, expected, videos = inputs(builder)
    for video in videos.values():
        path = (DATA / video['relative_path']).resolve()
        assert path.is_relative_to(DATA.resolve())
        assert path.stat().st_size == video['bytes'] and sha(path) == video['sha256'], path
    if args.cpu_check:
        cfg = env['tiny_config'](builder.processor)
        cfg['vision_config']['depth'] = 24
        model = env['Qwen3_5ForConditionalGeneration'](
            env['Qwen3_5Config'](**cfg, attn_implementation='sdpa')).float()
    else:
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            weights, local_files_only=True, dtype=torch.float32, device_map='cuda',
            attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    model.eval().requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    scorers = {}
    for arm, path in CHECKPOINTS.items():
        scorer = GroupScorer(model.config.vision_config.hidden_size).to(device)
        state = torch.load(ROOT / path, map_location='cpu', weights_only=True)
        assert sum(p.numel() for p in state.values()) == 69697
        assert all(torch.isfinite(p).all() for p in state.values())
        if not args.cpu_check:
            scorer.load_state_dict(state)
        scorers[arm] = scorer.eval().requires_grad_(False)
    args.out.mkdir()
    started = time.monotonic()
    total_pairs = total_answers = generated = 0
    # Preflight one real query with an actual history merge and all four QA paths.
    if args.cpu_check:
        _, vid, query = min(
            ((expected[q['question_id']]['pairs'], vid, q) for vid, qs in groups for q in qs
             if 17 <= expected[q['question_id']]['pairs'] <= 20),
            key=lambda entry: (entry[0], entry[1], entry[2]['question_id']))
        groups = [(vid, [query])]
    with torch.inference_mode(), env['fp32_reference'](), (args.out / 'answers.jsonl').open('x', encoding='utf-8') as log:
        for index, (vid, questions) in enumerate(groups):
            observer = Observer(model, builder, scorers)
            def append(first, second, *, padding):
                if time.monotonic() - started > 177 * 60:
                    raise TimeoutError('fixed evaluation deadline')
                observer.append(first, second, padding=padding)
            driver = PairDriver(CausalSource(DATA / videos[vid]['relative_path']), append)
            try:
                for q in questions:
                    checkpoint = driver.advance(q['question_time'])
                    target = expected[q['question_id']]
                    assert all(checkpoint[k] == target[k] for k in PREFIX_FIELDS), q['question_id']
                    offset = (total_answers // len(ARMS)) % len(ARMS)
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        bank = observer.banks[arm]
                        row = answer(env, model, builder, bank, prompt(q), limit=2 if args.cpu_check else 16)
                        assert row['visual_tokens'] <= 864 and all(
                            node['end'] <= q['question_time'] for node in row['nodes'])
                        if args.cpu_check:
                            assert bank.long_merges + bank.within_merges > 0
                        log.write(json.dumps(dict(question_id=q['question_id'], video_id=vid,
                                                  arm=arm, prefix=checkpoint, **row)) + '\n')
                        total_answers += 1
                        generated += len(row['generated_ids'])
                    log.flush()
                    assert (args.out / 'answers.jsonl').stat().st_size < 64 * 2**20
                total_pairs += observer.pairs
            finally:
                driver.close()
            print(json.dumps(dict(sources=index+1, answers=total_answers, pairs=total_pairs,
                                  seconds=time.monotonic()-started)), flush=True)
    if not args.cpu_check:
        assert (total_pairs, total_answers) == (14678, 1912)
    save(args.out / 'completed.json', dict(
        status='cpu_passed' if args.cpu_check else 'completed', checked_questions=478, checked_media=128,
        sources=len(groups), pairs=total_pairs, answers=total_answers, generated_tokens=generated,
        visual_path_evaluations=total_pairs*4, visual_prefix_evaluations=total_pairs*2,
        base_updates=0, scorer_updates=0, seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-check', action='store_true')
    main(parser.parse_args())
