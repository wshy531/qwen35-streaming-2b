"""Frozen four-way short-clip LS completion; no fitting or new data."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/2b-training-order-main-v2'))
import run_main as m
from resolution_banks_base import Bank
from resolution_nodes import digest
from adapter import sparse_pair
sys.path.insert(0, str(ROOT/'experiments/2b-content-midprune-v1'))
from content_prune import encode as fixed_encode
import torch

PREP = ROOT/'runs/short_ls_completion_prepare_20261104_01'
CHECKPOINTS = {
    'initial': 'runs/training-order-main-220483/initial.pt',
    'old_CE': 'runs/hard-mask-main-220062/mask_ce_step1237.pt',
    'guarded': 'runs/repair-guard-220499/final.pt',
}
ARMS = (*CHECKPOINTS, 'fixed12')


@torch.no_grad()
def fixed_nodes(model, pairs):
    device = model.get_input_embeddings().weight.device
    bank = Bank('LS', 54, budget=864, recent=432, closed=324)
    for i, (begin, end, pixels, grid) in enumerate(pairs):
        values, ids, _ = fixed_encode(model.model.visual, pixels.to(device), grid.to(device), mode='mid_fixed')
        assert ids.tolist() == list(range(1, 108, 2))
        bank.append(sparse_pair(values, ids, i, begin, end), starts_event=i % 3 == 0)
    return bank.export()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    assert not args.out.exists() and args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    for path, expected in m.read(PREP/'execution_lock.json').items():
        assert m.sha(ROOT/path) == expected, path
    groups = m.read(PREP/'short_manifest.json')
    assert len(groups) == 40 and sum(len(g['questions']) for g in groups) == 345
    assert sum(g['audit']['encoding_pairs'] for g in groups) == 565
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
        assert torch.cuda.device_count() == 1
    env = m.runtime()
    builder = m.make_builder(env, env['AutoProcessor'].from_pretrained(m.WEIGHTS, local_files_only=True))
    if args.cpu_check:
        cfg = env['tiny_config'](builder.processor)
        cfg['vision_config']['depth'] = 24
        model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg, attn_implementation='sdpa')).float()
    else:
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(m.WEIGHTS, local_files_only=True,
            dtype=torch.float32, device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    model.eval().requires_grad_(False)
    versions = {n: p._version for n, p in model.named_parameters()}
    scorer = m.GroupScorer(model.config.vision_config.hidden_size).to(next(model.parameters()).device)
    scorer.eval().requires_grad_(False)
    args.out.mkdir()
    start = time.monotonic()
    if args.cpu_check:
        layouts = maximum = 0
        width = model.get_input_embeddings().embedding_dim
        for group in groups:
            assert m.sha(ROOT/group['path']) == group['sha256']
            bank = Bank('LS', 54, budget=864, recent=432, closed=324)
            for i in range(group['audit']['encoding_pairs']):
                bank.append(sparse_pair(torch.zeros(54, width), torch.arange(1,108,2), i, 2.*i, 2.*i+1), starts_event=i%3==0)
            for q in group['questions']:
                for shift in (0,2):
                    _,_,ids,_ = m.context(model,builder,bank.export(),m.question_text(m.rotate(q,shift)),visual_limit=864)
                    layouts += 1
                    maximum = max(maximum,ids.numel())
        group = max(groups,key=lambda g:g['audit']['encoding_pairs'])
        pairs = m.load_pairs(env,builder,group)
        with torch.no_grad(), env['fp32_reference']():
            c = m.capture(model,scorer,pairs,False)
            manual = Bank('LS',54,budget=864,recent=432,closed=324)
            for i,node in enumerate(m.all_nodes(c,pairs)):
                manual.append(node,starts_event=i%3==0)
            assert digest(c.nodes)==digest(manual.export())
            fixed = fixed_nodes(model,pairs)
            for nodes in (c.nodes,fixed):
                for shift in (0,2):
                    m.answer(model,builder,group['questions'][0],nodes,shift)
        m.save(args.out/'completed.json',dict(status='cpu_passed',shared_layouts=layouts,arm_layouts=4*layouts,
            max_tokens=maximum,source_SHA=40,real_pairs=len(pairs),micro_answers=4,manual_bank_parity=True))
        return
    references = m.read(PREP/'references.json')
    counts = dict(sources=0,answers=0,regressions=0,visual_calls=0)
    def guard():
        if time.monotonic()-start>58*60:
            raise TimeoutError('fixed 58 minute stop; retain partial results')
        assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file()) < 96*2**20
    with torch.no_grad(), env['fp32_reference'](), (args.out/'answers.jsonl').open('x',encoding='utf-8') as log:
        for group in groups:
            guard()
            pairs = m.load_pairs(env,builder,group)
            for arm in ARMS:
                guard()
                if arm == 'fixed12':
                    nodes = fixed_nodes(model,pairs)
                else:
                    scorer.load_state_dict(torch.load(ROOT/CHECKPOINTS[arm],map_location='cpu',weights_only=True))
                    captured = m.capture(model,scorer,pairs,False)
                    nodes = captured.nodes
                    complete = m.all_nodes(captured,pairs)
                    # Reproduce every existing complete-short endpoint before
                    # publishing the LS readout for this source and checkpoint.
                    for q in group['questions']:
                        for shift in (0,2):
                            r = m.answer(model,builder,q,complete,shift)
                            old = references[arm][q['id']][str(shift)]
                            assert r['choice_original']==old['choice_original']
                            assert abs(r['ce']-old['ce'])<=1e-5+1e-5*abs(old['ce'])
                            log.write(json.dumps(dict(r,arm=arm,view='full_short',phase='regression'))+'\n')
                            counts['regressions'] += 1
                    del complete,captured
                counts['visual_calls'] += len(pairs)
                before = digest(nodes)
                assert sum(len(n.features) for n in nodes)<=864
                for q in group['questions']:
                    for shift in (0,2):
                        r = m.answer(model,builder,q,nodes,shift)
                        log.write(json.dumps(dict(r,arm=arm,view='LS',phase='evaluation',
                            visual_tokens=sum(len(n.features) for n in nodes)))+'\n')
                        counts['answers'] += 1
                assert before==digest(nodes)
                log.flush()
                del nodes
            counts['sources'] += 1
            del pairs
            gc.collect()
            print(json.dumps(dict(**counts,seconds=time.monotonic()-start)),flush=True)
    assert counts==dict(sources=40,answers=2760,regressions=2070,visual_calls=2260)
    assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
    m.save(args.out/'completed.json',dict(status='completed',**counts,questions=345,updates=0,
        generated_tokens=0,base_unchanged=True,seconds=time.monotonic()-start))


if __name__=='__main__':
    main()
