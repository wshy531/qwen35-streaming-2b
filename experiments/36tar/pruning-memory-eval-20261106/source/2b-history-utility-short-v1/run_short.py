"""Frozen short development transfer and full unpruned reference."""
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

PREP = ROOT / 'runs/history_utility_short_prepare_20261104_01'


@torch.no_grad()
def complete_answer(model, builder, nodes, question, shift):
    rotated = gate.m.rotate(question, shift)
    embeddings, positions, ids, _ = gate.m.context(
        model, builder, nodes, gate.m.question_text(rotated), visual_limit=2160)
    logits = gate.m.last_logits(model, embeddings, positions)
    options = gate.m.option_ids(builder, logits.device)
    choice = int(logits[0, options].argmax())
    return dict(id=question['id'], video_id=question['video_id'], type=question['type'],
        shift=shift, correct=choice == rotated['answer_index'], choice_original=(choice+shift)%5,
        ce=float(-logits[0].log_softmax(-1)[options[rotated['answer_index']]]),
        input_tokens=ids.numel(), visual_tokens=sum(len(n.features) for n in nodes), phase='full_short')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    assert not args.out.exists() and args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    lock = gate.m.read(PREP/'execution_lock.json')
    for path, digest in lock.items():
        assert gate.m.sha(ROOT/path) == digest, path
    groups = gate.m.read(PREP/'short_manifest.json')
    assert len(groups) == 40 and sum(len(g['questions']) for g in groups) == 345
    assert sum(g['audit']['encoding_pairs'] for g in groups) == 565
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] > 20*2**30
    args.out.mkdir()
    start = time.monotonic()
    env = gate.m.runtime()
    builder = gate.m.make_builder(env, env['AutoProcessor'].from_pretrained(gate.m.WEIGHTS, local_files_only=True))
    if args.cpu_check:
        config = env['tiny_config'](builder.processor)
        config['vision_config']['depth'] = 24
        model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config, attn_implementation='sdpa')).float()
    else:
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(gate.m.WEIGHTS,
            local_files_only=True, dtype=torch.float32, device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    model.eval().requires_grad_(False)
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    if args.cpu_check:
        layouts = 0
        maximum = 0
        width = model.get_input_embeddings().embedding_dim
        with env['fp32_reference']():
            for group in groups:
                assert gate.m.sha(ROOT/group['path']) == group['sha256']
                bank = gate.synthetic_bank(group, width)
                assert len(bank.history) == len(bank.recent) == 4
                complete = [gate.pair_node(torch.zeros(108, width), i, float(2*i), float(2*i+1), (9, 12))
                    for i in range(group['audit']['encoding_pairs'])]
                for question in group['questions']:
                    for nodes in (complete, bank.nodes(), [e.node for e in bank.recent]):
                        for shift in (0, 2):
                            _, _, ids, _ = gate.m.context(model, builder, nodes,
                                gate.m.question_text(gate.m.rotate(question, shift)), visual_limit=2160)
                            layouts += 1
                            maximum = max(maximum, ids.numel())
            longest = max(groups, key=lambda g:g['audit']['encoding_pairs'])
            pairs = gate.m.load_pairs(env, builder, longest)
            nodes = [gate.pair_node(gate.project(model, pair), i, pair[0], pair[1], (9, 12)) for i, pair in enumerate(pairs)]
            for shift in (0, 2):
                complete_answer(model, builder, nodes, longest['questions'][0], shift)
        gate.m.save(args.out/'completed.json', dict(status='cpu_passed', endpoint_layouts=layouts,
            source_SHA=40, max_tokens=maximum, real_visual_pairs=len(pairs), micro_answers=2))
        return
    counts = dict(sources=0, questions=0, classifications=0, representation_forwards=0, visual_calls=0)

    def guard():
        if time.monotonic()-start > 88*60:
            raise TimeoutError('Fixed 88 minute limit')
        assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file()) < 96*2**20

    with env['fp32_reference'](), (args.out/'answers.jsonl').open('x', encoding='utf-8') as log:
        for group in groups:
            guard()
            pairs = gate.m.load_pairs(env, builder, group)
            bank = gate.Bank('LS', 108)
            complete = []
            for index, pair in enumerate(pairs):
                node = gate.pair_node(gate.project(model, pair), index, pair[0], pair[1], (9, 12))
                complete.append(node)
                bank.append(node, starts_event=index%3 == 0)
                counts['visual_calls'] += 1
            history = [e.node for e in bank.history]
            recent = [e.node for e in bank.recent]
            before = gate.digest(bank.nodes())
            assert len(history) == len(recent) == 4
            source_features = []
            for question in group['questions']:
                guard()
                for shift in (0, 2):
                    log.write(json.dumps(complete_answer(model, builder, complete, question, shift))+'\n')
                    counts['classifications'] += 1
                for subset, choice in enumerate(gate.subsets(4)):
                    for shift in (0, 2):
                        row = gate.answer(model, builder, gate.selected_nodes(recent, history, choice), question, shift)
                        log.write(json.dumps(dict(row, phase='fit', subset=subset, history_indices=choice))+'\n')
                        counts['classifications'] += 1
                source_features.append(gate.features(model, builder, recent, history, question))
                counts['representation_forwards'] += 1
                counts['questions'] += 1
                log.flush()
            assert gate.digest(bank.nodes()) == before
            torch.save(dict(rows=source_features, source_sha256=group['sha256'], bank_sha256=before,
                history=[n.metadata() for n in history], recent=[n.metadata() for n in recent]),
                args.out/('features-'+group['video_id']+'.pt'))
            counts['sources'] += 1
            print(json.dumps(dict(**counts, seconds=time.monotonic()-start)), flush=True)
            del pairs, complete, bank, history, recent, source_features
            gc.collect()
    assert counts == dict(sources=40, questions=345, classifications=11730, representation_forwards=345, visual_calls=565)
    assert all(p._version == versions[n] and p.grad is None for n, p in model.named_parameters())
    gate.m.save(args.out/'completed.json', dict(status='completed', **counts, seconds=time.monotonic()-start,
        base_unchanged=True, updates=0, generated_tokens=0, full_short_storage='finite reference only; not a bounded streaming claim'))


if __name__ == '__main__':
    main()
