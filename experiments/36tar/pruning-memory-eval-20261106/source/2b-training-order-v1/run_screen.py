"""Bounded two-arm screening on a frozen, source-separated training panel."""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from run_order import (
    ROOT, WEIGHTS, GroupScorer, capture_pairs, context, last_logits, loss_parts,
    make_builder, option_ids, prepare_inputs, qa_leaf_gradients, question_text,
    read, rotate, runtime, schedules, sha, snapshot, train_shift, visual_backward,
)

PREP = ROOT / 'runs/training_order_screen_20261104_01'


def read_nodes(capture, row, count, kind):
    recent = [node for node in capture.nodes if node.first >= max(0, count - 8)]
    assert len(recent) == min(8, count) and all(n.first == n.last for n in recent)
    nodes = recent if kind == 'short' else capture.nodes
    evidence = row['evidence_pairs']
    assert all(any(n.first <= i <= n.last for n in nodes) for i in evidence)
    if kind == 'delayed':
        assert max(evidence) < count - 8
    merged = [i for i in evidence if any(n.first <= i <= n.last and n.first < n.last for n in nodes)]
    return nodes, recent, merged


def train_sample(model, builder, scorer, inputs, row, kind):
    prefix = inputs[row['video_id']][:row[kind + '_pair_count']]
    visual = model.model.visual
    capture = capture_pairs(visual, scorer, prefix)
    capture.nodes, _, merged = read_nodes(capture, row, len(prefix), kind)
    question = rotate(row['question'], train_shift(row['question']))
    qa = qa_leaf_gradients(model, builder, capture, question_text(question), question['answer_index'])
    scaled = {index: value * .5 for index, value in qa['gradients'].items()}
    scores = visual_backward(visual, scorer, capture, scaled, prefix.__getitem__)
    evidence = row['evidence_pairs']
    assert all(i in scores for i in evidence)
    return dict(id=row['id'], kind=kind, ce=qa['loss'], merged_evidence_pairs=merged,
                evidence_grad_l1=sum(float(scores[i].abs().sum()) for i in evidence),
                merged_evidence_grad_l1=sum(float(scores[i].abs().sum()) for i in merged),
                visual_calls=len(prefix)+len(scaled))


@torch.no_grad()
def score(model, builder, row, nodes, shift, readout, merged):
    question = rotate(row['question'], shift)
    prompt = question_text(question)
    device = model.get_input_embeddings().weight.device
    if nodes is None:
        rendered = builder.template(prompt)
        assert rendered.count(builder.marker) == 1
        ids = builder.tokenize(rendered.replace(builder.marker, '', 1)).to(device)
        assert not (ids == builder.processor.video_token_id).any()
        emb = model.get_input_embeddings()(ids)
        pos = torch.arange(ids.shape[1], device=device).expand(4, 1, -1)
    else:
        emb, pos, _, _ = context(model, builder, nodes, prompt)
    logits = last_logits(model, emb, pos)
    options = option_ids(builder, device)
    choice = int(logits[0, options].argmax())
    answer = question['answer_index']
    return dict(id=row['id'], video_id=row['video_id'], prior_preflight=row['prior_preflight'],
                cue_group=row['cue_group'], shift=shift, readout=readout,
                correct=choice==answer, choice_original=(choice+shift)%5,
                ce=float(loss_parts(logits, options, answer)[1]), merged_evidence_pairs=merged)


def evaluate(model, builder, scorer, inputs, rows, controls):
    results = []
    visual_calls = 0
    for row in rows:
        for kind in ('short', 'delayed'):
            prefix = inputs[row['video_id']][:row[kind + '_pair_count']]
            capture = capture_pairs(model.model.visual, scorer, prefix)
            visual_calls += len(prefix)
            nodes, recent, merged = read_nodes(capture, row, len(prefix), kind)
            for shift in (0, 2):
                results.append(score(model, builder, row, nodes, shift, kind, merged))
                if controls and kind == 'delayed':
                    results.append(score(model, builder, row, recent, shift, 'late_recent', []))
    return dict(answers=results, visual_calls=visual_calls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-tiny', action='store_true')
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    for path, digest in read(PREP/'execution_lock.json').items():
        assert sha(ROOT/path) == digest, path
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_tiny:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1
    panel = read(PREP/'panel_LOCAL_ONLY.json')
    train = {r['id']: r for r in panel['train']}
    development = panel['development']
    assert len(train) == 8 and len(development) == 4
    assert {r['video_id'] for r in train.values()}.isdisjoint(r['video_id'] for r in development)
    plan = schedules(list(train), list(train))
    args.out.mkdir()
    started = time.monotonic()
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(WEIGHTS, local_files_only=True)
    builder = make_builder(env, processor)
    inputs = prepare_inputs([*train.values(), *development], env, builder)
    if args.cpu_tiny:
        config = env['tiny_config'](processor)
        config['vision_config']['depth'] = 24
        model = env['Qwen3_5ForConditionalGeneration'](
            env['Qwen3_5Config'](**config, attn_implementation='sdpa')).float()
    else:
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            WEIGHTS, local_files_only=True, dtype=torch.float32, device_map='cuda',
            attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    model.eval().requires_grad_(False)
    versions = {name: p._version for name, p in model.named_parameters()}
    device = model.get_input_embeddings().weight.device
    torch.manual_seed(20261104)
    scorer = GroupScorer(model.config.vision_config.hidden_size).to(device)
    initial = snapshot(scorer)
    torch.save(initial, args.out/'initial.pt')
    updates = []
    evaluations = {}
    with env['fp32_reference'](), (args.out/'updates.jsonl').open('x', encoding='utf-8') as log:
        text = [score(model, builder, row, None, shift, 'text', [])
                for row in development for shift in (0, 2)]
        evaluations['initial'] = evaluate(model, builder, scorer, inputs, development, controls=True)
        for arm, batches in plan.items():
            scorer.load_state_dict(initial)
            optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-4, weight_decay=0.)
            for step, batch in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                assert all(item['loss_scale'] == .5 for item in batch)
                samples = [train_sample(model, builder, scorer, inputs, train[item['id']], item['kind'])
                           for item in batch]
                norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1., error_if_nonfinite=True)
                assert float(norm) > 0
                optimizer.step()
                peak = torch.cuda.max_memory_allocated() if device.type=='cuda' else None
                report = dict(arm=arm, step=step, samples=samples, grad_norm=float(norm), peak_allocated_bytes=peak)
                updates.append(report)
                log.write(json.dumps(report)+'\n')
                log.flush()
                if step % 2 == 0:
                    state = snapshot(scorer)
                    assert all(torch.isfinite(value).all() for value in state.values())
                    torch.save(state, args.out/f'{arm}_step{step}.pt')
                    evaluations[f'{arm}_step{step}'] = evaluate(
                        model, builder, scorer, inputs, development, controls=step==len(batches))
            assert any(not torch.equal(value, initial[name]) for name, value in state.items())
    assert all(p.grad is None and p._version == versions[n] for n, p in model.named_parameters())
    train_rows = [sample for update in updates for sample in update['samples']]
    eval_rows = [row for value in evaluations.values() for row in value['answers']]
    visual_calls = sum(r['visual_calls'] for r in train_rows) + sum(v['visual_calls'] for v in evaluations.values())
    assert len(updates)==16 and len(train_rows)==32 and len(eval_rows)==168 and len(text)==8
    assert visual_calls<=2110
    report = dict(status='completed', cpu_tiny=args.cpu_tiny, updates=len(updates),
                  language_forwards=len(train_rows)+len(eval_rows)+len(text), visual_calls=visual_calls,
                  evaluations=evaluations, text=text, seconds=time.monotonic()-started, generated_tokens=0)
    (args.out/'completed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('evaluations','text')}), flush=True)


if __name__ == '__main__':
    main()
