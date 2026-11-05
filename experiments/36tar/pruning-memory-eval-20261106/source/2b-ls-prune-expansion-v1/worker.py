"""Causal bounded LS evaluation using the previously frozen video inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for experiment in ('streamingbench-router-v1', '2b-midprune-memory-v1'):
    sys.path.insert(0, str(ROOT / 'experiments' / experiment))
from v0_source import V0Pairs
from pilot import (P, Memory, VIEWS, GroupScorer, CoverageScorer, exact_pairs,
                   runtime, make_builder, hard_encode, encode_pair, question, read, sha, torch)
from adapter import Bank, THRESHOLD, ask, context, distance, sparse_pair
from response_v4 import parse

OUT = ROOT / 'runs/ls_prune_expansion_prepare_20261103_01'
AUDIT = ROOT / 'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json'
REF = ROOT / 'runs/protection_streamingbench_prepare_20261103_01/causal_reference.json'
OLD = ROOT / 'runs/ls-prune-coverage-220370/answers.jsonl'
PANEL = OUT / 'regression300.json'
CAPACITY = ((4, 4), (4, 8), (8, 4), (8, 8))
MINUTES = 480


def groups():
    result = defaultdict(list)
    for q in read(PANEL):
        result[q['dataset'], q['source_id']].append(q)
    return [(key, sorted(qs, key=lambda q: (q['query_time_s'], q['id'])))
            for key, qs in sorted(result.items())]


def inputs():
    return ({r['video_id']: r for r in read(AUDIT)['rows']}, read(REF)['prefixes'])


def ls_only(view, r=None, h=None):
    mem = Memory(view)
    if r is not None:
        mem.banks['LS'] = Bank('LS', 54, budget=54*(r+h), recent=54*r, closed=162)
    mem.banks = {'LS': mem.banks['LS']}
    return mem


class Observe:
    def __init__(self, model, head, capacity=False):
        self.model, self.head, self.capacity = model, head, capacity
        self.selector, self.previous = CoverageScorer(), None
        self.readers = {v: ls_only(v) for v in VIEWS} if not capacity else {
            f'{v}_R{r}H{h}': ls_only(v, r, h) for v in VIEWS[1:] for r, h in CAPACITY}
        self.descriptors = {}
        self.index = 0

    def append(self, batch, begin, end):
        batch = batch.to('cuda')
        for view in VIEWS[1:] if self.capacity else VIEWS:
            if view == 'mask_ce':
                features, ids, _, self.previous = hard_encode(self.model.model.visual,
                    batch.pixel_values_videos, batch.video_grid_thw, self.head,
                    previous=self.previous, mode='gather')
            else:
                features, ids, _ = encode_pair(self.model, batch, view, None, None, self.selector)
            if not self.capacity:
                self.readers[view].append(features, ids, self.index, begin, end)
                continue
            node = sparse_pair(features, ids, self.index, begin, end)
            desc = node.features.mean(0)
            previous = self.descriptors.get(view)
            starts = previous is None or distance(previous, desc) > THRESHOLD
            self.descriptors[view] = desc
            for r, h in CAPACITY:
                self.readers[f'{view}_R{r}H{h}'].banks['LS'].append(node, starts_event=starts)
        self.index += 1


def verify_lock():
    for name, expected in read(OUT / 'execution_lock.json').items():
        path = ROOT / name
        if name.startswith('repo/') and not path.exists():
            path = ROOT / 'publish/qwen35-streaming-2b' / name[5:]
        assert sha(path) == expected, name


def run(dest):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
    verify_lock()
    auth = read(OUT / 'GPU_authorization.json')
    assert auth['approved'] and auth['lock_sha256'] == sha(OUT / 'execution_lock.json')
    assert read(OUT / 'CPU_local.json') == read(OUT / 'CPU_cluster.json')
    assert read(OUT / 'media_cluster.json')['files'] == 227
    assert dest.resolve().is_relative_to((ROOT / 'runs').resolve())
    dest.mkdir()
    start = time.monotonic()
    counts = dict(main=0, capacity=0, replayed=0, visual_calls=0, generated_tokens=0)
    old = {}
    for line in OLD.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        old[row['id'], row['arm']] = row

    def guard():
        if time.monotonic()-start > (MINUTES-3)*60:
            raise TimeoutError('fixed internal stop line')
        assert sum(p.stat().st_size for p in dest.iterdir()) < 64*2**20

    def answer(q, observer, phase, stream):
        arms = list(observer.readers)
        offset = counts[phase] // len(arms) % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            guard()
            reader = observer.readers[arm]
            result = ask(env, model, builder, reader, 'LS', question(q, 0, 'B'), limit=128)
            assert all(node['end'] <= q['query_time_s']+1e-6 for node in result['nodes'])
            reference = None
            if phase == 'capacity':
                view, size = arm.rsplit('_', 1)
                if size in ('R4H4', 'R8H8'):
                    reference = old[q['id'], view + ('_auto432' if size == 'R4H4' else '_auto864')]
            if reference is not None:
                for key in ('generated_ids', 'input_ids_sha256', 'position_ids_sha256', 'nodes'):
                    assert result[key] == reference[key], (q['id'], arm, key)
                counts['replayed'] += 1
            result.update(parse(result['text'], len(q['options']), 'B'))
            result.update(id=q['id'], arm=arm, phase=phase, query_time_s=q['query_time_s'],
                          source_id=q.get('source_id', q.get('source_name')),
                          dataset=q.get('dataset', 'S-EMBER-diagnostic'))
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            stream.flush()
            counts[phase] += 1
            counts['generated_tokens'] += len(result['generated_ids'])

    try:
        torch.set_num_threads(1)
        env = runtime()
        assert torch.cuda.device_count() == 1 and env['psutil'].virtual_memory().available >= 12*2**30
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            ROOT/'models/Qwen3.5-2B', local_files_only=True, dtype=torch.float32,
            device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False)
        builder = make_builder(env, env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B', local_files_only=True))
        head = GroupScorer(model.config.vision_config.hidden_size).to('cuda').eval().requires_grad_(False)
        spec = read(P/'protocol.json')
        head.load_state_dict(torch.load(ROOT/spec['checkpoint'], map_location='cpu', weights_only=True))
        audits, prefixes = inputs()
        with torch.no_grad(), env['fp32_reference'](), (dest/'answers.jsonl').open('x', encoding='utf-8') as stream:
            for source in read(P/'sources.json'):
                obs = Observe(model, head, capacity=True)
                tasks = [q for q in read(P/'questions.json') if q['source_name'] == source['name']]
                delivered = set()
                path = Path(spec['media_root']) / source['name']
                assert sha(path) == source['media_sha256']
                for first, second in exact_pairs(env, path, source['sampled_trace']):
                    guard()
                    batch, _ = builder.pair(first, second, 0, 0)
                    obs.append(batch, first.timestamp, second.timestamp)
                    counts['visual_calls'] += 3
                    for q in tasks:
                        if q['query_time_s'] == second.timestamp:
                            assert q['id'] not in delivered
                            answer(q, obs, 'capacity', stream)
                            delivered.add(q['id'])
                assert delivered == {q['id'] for q in tasks}
            assert counts['capacity'] == 216 and counts['replayed'] == 108
            for (dataset, source_id), tasks in groups():
                guard()
                obs = Observe(model, head)
                path = ROOT/tasks[0]['media_path']
                assert sha(path) == tasks[0]['media_sha256']
                if dataset == 'NExT-QA':
                    pairs = env_decode(env, path, audits[source_id], builder)
                    for begin, end, batch in pairs:
                        guard()
                        obs.append(batch, begin, end)
                        counts['visual_calls'] += 4
                    for q in tasks:
                        answer(q, obs, 'main', stream)
                else:
                    def commit(first, second, padding):
                        guard()
                        batch, _ = builder.pair(first, second, 0, 0, synthetic_padding=padding)
                        obs.append(batch, first.timestamp, second.timestamp)
                        counts['visual_calls'] += 4

                    driver = V0Pairs(path, commit, config='auto4')
                    try:
                        for q in tasks:
                            state = driver.advance(q['query_time_s'])
                            assert state == prefixes[q['id']]['state'], q['id']
                            assert obs.index == state['complete_pairs']
                            answer(q, obs, 'main', stream)
                    finally:
                        driver.close()
                print(json.dumps(dict(dataset=dataset, source=source_id, seconds=time.monotonic()-start, **counts)), flush=True)
        assert counts['main'] == 1200 and counts['visual_calls'] == 108879
        assert counts['generated_tokens'] <= 1416*128
        verify_lock()
        (dest/'completed.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    except BaseException as exc:
        (dest/'failed.json').write_text(json.dumps(dict(error=repr(exc), **counts)), encoding='utf-8')
        raise


from protocol import decode as env_decode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
