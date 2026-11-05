"""Complete frozen utility supervision using the verified gate readout."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-history-utility-v1'))
import score_gate as gate
import torch

PREP = ROOT / 'runs/history_utility_main_prepare_20261104_01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['remaining_train', 'development'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    assert not args.out.exists() and args.out.resolve().is_relative_to((ROOT / 'runs').resolve())
    lock = gate.m.read(PREP / 'execution_lock.json')
    for path, digest in lock.items():
        assert gate.m.sha(ROOT / path) == digest, path
    groups = gate.m.read(PREP / (args.split + '.json'))
    expected = {'remaining_train': (88, 800, 4861), 'development': (30, 271, 1710)}[args.split]
    assert (len(groups), sum(len(g['questions']) for g in groups), sum(g['audit']['encoding_pairs'] for g in groups)) == expected
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] > 20 * 2**30
    args.out.mkdir()
    start = time.monotonic()
    env = gate.m.runtime()
    builder = gate.m.make_builder(env, env['AutoProcessor'].from_pretrained(gate.m.WEIGHTS, local_files_only=True))
    if args.cpu_check:
        config = env['tiny_config'](builder.processor)
        config['vision_config']['depth'] = 24
        model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config, attn_implementation='sdpa')).float()
    else:
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            gate.m.WEIGHTS, local_files_only=True, dtype=torch.float32,
            device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    model.eval().requires_grad_(False)
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    if args.cpu_check:
        layouts = 0
        maximum = 0
        with env['fp32_reference']():
            for group in groups:
                assert gate.m.sha(ROOT / group['path']) == group['sha256']
                bank = gate.synthetic_bank(group, model.get_input_embeddings().embedding_dim)
                recent = [entry.node for entry in bank.recent]
                assert len(bank.history) == 4 and len(recent) == 4
                for question in group['questions']:
                    for nodes in (recent, bank.nodes()):
                        for shift in (0, 2):
                            _, _, ids, _ = gate.m.context(model, builder, nodes,
                                gate.m.question_text(gate.m.rotate(question, shift)), visual_limit=864)
                            layouts += 1
                            maximum = max(maximum, ids.numel())
                    gate.m.context(model, builder, recent, question['question'], visual_limit=864)
            pairs = gate.m.load_pairs(env, builder, groups[0])[:10]
            bank = gate.Bank('LS', 108)
            for index, pair in enumerate(pairs):
                bank.append(gate.pair_node(gate.project(model, pair), index, pair[0], pair[1], (9, 12)), starts_event=index % 3 == 0)
            recent = [entry.node for entry in bank.recent]
            history = [entry.node for entry in bank.history]
            before = gate.digest(bank.nodes())
            for choice in ((), (0,), (0, 1, 2, 3)):
                gate.answer(model, builder, gate.selected_nodes(recent, history, choice), groups[0]['questions'][0], 0)
            gate.features(model, builder, recent, history, groups[0]['questions'][0])
            assert gate.digest(bank.nodes()) == before
        gate.m.save(args.out / 'completed.json', dict(status='cpu_passed', split=args.split,
            endpoint_layouts=layouts, query_layouts=expected[1], max_tokens=maximum,
            source_SHA=expected[0], real_pairs=10, micro_answers=3))
        return
    counts = dict(sources=0, questions=0, classifications=0, representation_forwards=0, visual_calls=0)
    limit = 137 * 60 if args.split == 'remaining_train' else 67 * 60

    def guard():
        if time.monotonic() - start > limit:
            raise TimeoutError('Fixed phase limit reached')
        assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file()) < 160 * 2**20

    with env['fp32_reference'](), (args.out / 'answers.jsonl').open('x', encoding='utf-8') as log:
        for group in groups:
            guard()
            pairs = gate.m.load_pairs(env, builder, group)
            bank = gate.Bank('LS', 108)
            for index, pair in enumerate(pairs):
                guard()
                bank.append(gate.pair_node(gate.project(model, pair), index, pair[0], pair[1], (9, 12)), starts_event=index % 3 == 0)
                counts['visual_calls'] += 1
            history = [entry.node for entry in bank.history]
            recent = [entry.node for entry in bank.recent]
            before = gate.digest(bank.nodes())
            assert len(history) == 4 and len(recent) == 4
            source_features = []
            for question in group['questions']:
                guard()
                for subset, choice in enumerate(gate.subsets(4)):
                    for shift in (0, 2):
                        row = gate.answer(model, builder, gate.selected_nodes(recent, history, choice), question, shift)
                        log.write(json.dumps(dict(row, phase='fit', subset=subset, history_indices=choice)) + '\n')
                        counts['classifications'] += 1
                source_features.append(gate.features(model, builder, recent, history, question))
                counts['representation_forwards'] += 1
                counts['questions'] += 1
                log.flush()
            assert gate.digest(bank.nodes()) == before
            torch.save(dict(rows=source_features, history=[n.metadata() for n in history],
                recent=[n.metadata() for n in recent], bank_sha256=before, source_sha256=group['sha256']),
                args.out / ('features-' + group['video_id'] + '.pt'))
            counts['sources'] += 1
            print(json.dumps(dict(**counts, seconds=time.monotonic() - start)), flush=True)
            del source_features, pairs, bank, history, recent
            gc.collect()
    assert counts == dict(sources=expected[0], questions=expected[1], classifications=32*expected[1],
        representation_forwards=expected[1], visual_calls=expected[2])
    assert all(p._version == versions[n] and p.grad is None for n, p in model.named_parameters())
    for path, digest in lock.items():
        assert gate.m.sha(ROOT / path) == digest, path
    gate.m.save(args.out / 'completed.json', dict(status='completed', **counts,
        seconds=time.monotonic()-start, base_unchanged=True, generated_tokens=0, updates=0))


if __name__ == '__main__':
    main()
