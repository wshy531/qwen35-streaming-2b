"""Same-initialization curriculum versus mixed training on paired QA prefixes."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'runs/training_order_profile_20261104_01'
for name in ('2b-delayed-readiness-v1', '2b-delayed-recompute-v2'):
    sys.path.insert(0, str(ROOT / 'experiments' / name))
from profile_delayed_qa import WEIGHTS, GroupScorer, decode, make_builder, question_text, read, runtime, sha
from same_device import capture_pairs, qa_leaf_gradients, visual_backward
from delayed_qa import last_logits
from objective import loss_parts, option_ids
from protocol import rotate, train_shift
from resolution_runtime import context
from training_order import schedules


def snapshot(scorer):
    return {name: value.detach().cpu().clone() for name, value in scorer.state_dict().items()}


def prepare_inputs(rows, env, builder):
    result = {}
    for row in rows:
        source = row['source']
        if source['video_id'] in result:
            continue
        path = ROOT / 'datasets/delayed-training-nextqa-v1' / source['mirror_path']
        if sha(path) != source['mirror_lfs_sha256']:
            raise ValueError('Media SHA mismatch')
        pairs = decode(env, path, {'samples': row['samples']}, builder)
        if len(pairs) != 24:
            raise ValueError('Expected 24 real pairs')
        result[source['video_id']] = [
            (start, end, batch.pixel_values_videos, batch.video_grid_thw)
            for start, end, batch in pairs
        ]
    return result


def evidence_status(capture, evidence, count, kind):
    spans = [[node.first, node.last] for node in capture.nodes]
    covering = [span for span in spans if any(span[0] <= i <= span[1] for i in evidence)]
    if not all(any(a <= i <= b for a, b in covering) for i in evidence):
        raise RuntimeError('Evidence pair missing from bank')
    compressed = [i for i in evidence if any(a <= i <= b and a < b for a, b in covering)]
    if kind == 'delayed':
        if max(evidence) >= count - 8 or not compressed:
            raise RuntimeError('Delayed evidence did not enter a merged history node')
    elif covering != [[i, i] for i in evidence] or min(evidence) < max(0, count - 8):
        raise RuntimeError('Short evidence must remain recent and unmerged')
    return dict(covering_spans=covering, compressed_evidence_pairs=compressed)


def example(model, builder, scorer, inputs, row, kind, train):
    count = row[kind + '_pair_count']
    prefix = inputs[:count]
    question = rotate(row['question'], train_shift(row['question']))
    prompt = question_text(question)
    answer = question['answer_index']
    visual = model.model.visual
    capture = capture_pairs(visual, scorer, prefix)
    record = evidence_status(capture, row['evidence_pairs'], count, kind)
    device = next(scorer.parameters()).device
    if train:
        qa = qa_leaf_gradients(model, builder, capture, prompt, answer)
        gradients = {index: value * .5 for index, value in qa['gradients'].items()}
        score_gradients = visual_backward(visual, scorer, capture, gradients, prefix.__getitem__)
        checked_pairs = record['compressed_evidence_pairs'] if kind == 'delayed' else row['evidence_pairs']
        evidence_grad = sum(float(score_gradients[i].abs().sum()) for i in checked_pairs)
        if evidence_grad <= 0:
            raise RuntimeError('No score gradient on checked evidence')
        ce, logits, visual_calls = qa['loss'], qa['logits'], count + len(gradients)
        record['evidence_score_grad_l1_scaled'] = evidence_grad
    else:
        with torch.no_grad():
            emb, pos, _, _ = context(model, builder, capture.nodes, prompt)
            logits = last_logits(model, emb, pos)
            ce = float(loss_parts(logits, option_ids(builder, device), answer)[0])
        visual_calls = count
    choice = int(logits[0, option_ids(builder, logits.device)].argmax())
    return dict(record, id=row['id'], kind=kind, pairs=count, ce=ce,
                correct=choice == answer, option_argmax=choice, visual_calls=visual_calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-tiny', action='store_true')
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('Output must be inside project runs')
    for name, digest in read(PREP / 'execution_lock.json').items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Execution lock mismatch: {name}')
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_tiny:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1
    data = read(PREP / 'panel_LOCAL_ONLY.json')
    train_rows = {row['id']: row for row in data['train']}
    development = data['development']
    assert len(train_rows) == len(development) == 2
    assert {r['video_id'] for r in train_rows.values()}.isdisjoint(r['video_id'] for r in development)
    plan = schedules(list(train_rows), list(train_rows))
    args.out.mkdir()
    started = time.monotonic()
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(WEIGHTS, local_files_only=True)
    builder = make_builder(env, processor)
    inputs = prepare_inputs([*train_rows.values(), *development], env, builder)
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
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    device = model.get_input_embeddings().weight.device
    torch.manual_seed(20261104)
    scorer = GroupScorer(model.config.vision_config.hidden_size).to(device)
    initial = snapshot(scorer)
    torch.save(initial, args.out / 'initial.pt')
    evaluations = {}
    updates = []

    def evaluate(name):
        evaluations[name] = [example(model, builder, scorer, inputs[row['video_id']], row, kind, False)
                             for row in development for kind in ('short', 'delayed')]

    with env['fp32_reference'](), (args.out / 'updates.jsonl').open('x', encoding='utf-8') as log:
        evaluate('initial')
        for arm, batches in plan.items():
            scorer.load_state_dict(initial)
            optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-4, weight_decay=0.)
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                samples = []
                for item in batch:
                    assert item['loss_scale'] == .5
                    row = train_rows[item['id']]
                    samples.append(example(model, builder, scorer, inputs[row['video_id']], row, item['kind'], True))
                norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1., error_if_nonfinite=True)
                assert float(norm) > 0
                optimizer.step()
                state = snapshot(scorer)
                assert all(torch.isfinite(value).all() for value in state.values())
                torch.save(state, args.out / f'{arm}_step{step}.pt')
                report = dict(arm=arm, step=step, gradient_norm=float(norm), samples=samples,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
                updates.append(report)
                log.write(json.dumps(report)+'\n')
                log.flush()
                evaluate(f'{arm}_step{step}')
            assert any(not torch.equal(value, initial[name]) for name, value in state.items())
    assert all(p.grad is None and p._version == versions[n] for n, p in model.named_parameters())
    records = [s for update in updates for s in update['samples']]
    records.extend(row for rows in evaluations.values() for row in rows)
    assert len(updates) == 4 and len(records) == 28
    visual_calls = sum(row['visual_calls'] for row in records)
    assert visual_calls <= 520
    report = dict(status='completed', cpu_tiny=args.cpu_tiny, updates=len(updates),
                  language_forwards=len(records), visual_calls=visual_calls, evaluations=evaluations,
                  seconds=time.monotonic()-started, generated_tokens=0,
                  purpose='execution and evidence-gradient preflight, not a capability comparison')
    (args.out / 'completed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='evaluations'}), flush=True)


if __name__ == '__main__':
    main()
