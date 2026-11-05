"""Separate feature loss, merge scheduling and capacity in the frozen LS pilot."""
import json
import os
from pathlib import Path
import time

from matched_ls import MatchedLS, ROOT, Bank, sparse_pair, structure
from pilot import (P, Memory, VIEWS, GroupScorer, CoverageScorer, exact_pairs,
                   runtime, make_builder, hard_encode, encode_pair, question, read, sha, torch)
from adapter import THRESHOLD, ask, distance

OUT = ROOT / 'runs/ls_prune_coverage_prepare_20261103_01'
PRUNED = VIEWS[1:]


class BanksForReadout:
    def __init__(self):
        self.matched = MatchedLS(PRUNED)
        self.readers = {'full': self.wrap('full', self.matched.full)}
        self.previous = {}
        for view in PRUNED:
            self.readers[view + '_matched432'] = self.wrap(view, self.matched.pruned[view])
            for capacity, recent in ((432, 216), (864, 432)):
                self.readers[f'{view}_auto{capacity}'] = self.wrap(view, Bank('LS', 54, budget=capacity, recent=recent, closed=162))

    @staticmethod
    def wrap(view, bank):
        memory = Memory(view)
        memory.banks = {'LS': bank}
        return memory

    def append(self, nodes):
        flags = {}
        for view, node in nodes.items():
            desc = node.features.mean(0)
            previous = self.previous.get(view)
            flags[view] = previous is None or distance(previous, desc) > THRESHOLD
            self.previous[view] = desc
        choices = self.matched.append(nodes['full'], {v: nodes[v] for v in PRUNED}, starts_event=flags['full'])
        for view in PRUNED:
            for capacity in (432, 864):
                self.readers[f'{view}_auto{capacity}'].banks['LS'].append(nodes[view], starts_event=flags[view])
        return dict(starts_event=flags, full_merge_choices=choices)


def verify():
    for name, expected in read(OUT / 'execution_lock.json').items():
        p = ROOT / name
        if name.startswith('repo/') and not p.exists():
            p = ROOT / 'publish/qwen35-streaming-2b' / name[5:]
        assert sha(p) == expected, name


def run(destination):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
    verify()
    auth = read(OUT / 'GPU_authorization.json')
    assert auth['approved'] and auth['lock_sha256'] == sha(OUT / 'execution_lock.json')
    assert read(OUT / 'CPU_local.json') == read(OUT / 'CPU_cluster.json')
    assert read(OUT / 'media_cluster.json')['pixel_verified_pairs'] == 429
    assert destination.resolve().is_relative_to((ROOT / 'runs').resolve())
    destination.mkdir()
    start = time.monotonic()
    counts = dict(answers=0, replayed=0, visual_calls=0, generated_tokens=0)
    spec = read(P / 'protocol.json')
    questions = read(P / 'questions.json')
    expected = read(OUT / 'replay.json')

    def guard():
        if time.monotonic() - start > 33 * 60:
            raise TimeoutError('33 minute internal stop line')
        assert sum(p.stat().st_size for p in destination.iterdir() if p.is_file()) < 64 * 2**20

    try:
        env = runtime()
        torch.set_num_threads(1)
        assert torch.cuda.device_count() == 1 and env['psutil'].virtual_memory().available >= 12 * 2**30
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            ROOT / 'models/Qwen3.5-2B', local_files_only=True, dtype=torch.float32,
            device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        model.eval().requires_grad_(False)
        builder = make_builder(env, env['AutoProcessor'].from_pretrained(ROOT / 'models/Qwen3.5-2B', local_files_only=True))
        head = GroupScorer(model.config.vision_config.hidden_size).to('cuda').eval().requires_grad_(False)
        head.load_state_dict(torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=True))
        with (torch.no_grad(), env['fp32_reference'](),
              (destination / 'answers.jsonl').open('x', encoding='utf-8') as answers,
              (destination / 'schedule.jsonl').open('x', encoding='utf-8') as schedule):
            for source in read(P / 'sources.json'):
                banks = BanksForReadout()
                state = None
                selector = CoverageScorer()
                path = Path(spec['media_root']) / source['name']
                assert path.resolve().parent == Path(spec['media_root']).resolve() and sha(path) == source['media_sha256']
                tasks = [q for q in questions if q['source_name'] == source['name']]
                delivered = set()
                for i, (first, second) in enumerate(exact_pairs(env, path, source['sampled_trace'])):
                    guard()
                    batch, _ = builder.pair(first, second, 0, 0)
                    batch = batch.to('cuda')
                    nodes = {}
                    for view in VIEWS:
                        if view == 'mask_ce':
                            feats, ids, _, state = hard_encode(model.model.visual, batch.pixel_values_videos,
                                batch.video_grid_thw, head, previous=state, mode='gather')
                        else:
                            feats, ids, _ = encode_pair(model, batch, view, None, None, selector)
                        counts['visual_calls'] += 1
                        nodes[view] = sparse_pair(feats, ids, i, first.timestamp, second.timestamp)
                    trace = banks.append(nodes)
                    schedule.write(json.dumps(dict(source=source['name'], pair=i, **trace)) + '\n')
                    del nodes, batch, feats, ids
                    for q in tasks:
                        if q['query_time_s'] != second.timestamp:
                            continue
                        assert q['id'] not in delivered
                        delivered.add(q['id'])
                        order = ['full', *[v + '_auto864' for v in PRUNED]]
                        order += [a for a in banks.readers if a not in order]
                        for arm in order:
                            guard()
                            reader = banks.readers[arm]
                            result = ask(env, model, builder, reader, 'LS', question(q, 0, 'B'), limit=128)
                            if arm in expected[q['id']]:
                                for key, value in expected[q['id']][arm].items():
                                    assert result[key] == value, (q['id'], arm, key)
                                counts['replayed'] += 1
                            result.update(id=q['id'], arm=arm, role=q['role'], source=source['name'],
                                          temporal_structure=structure(reader.banks['LS']))
                            counts['answers'] += 1
                            counts['generated_tokens'] += len(result['generated_ids'])
                            answers.write(json.dumps(result, ensure_ascii=False) + '\n')
                            answers.flush()
                assert delivered == {q['id'] for q in tasks}
                print(json.dumps(dict(source=source['name'], **counts)), flush=True)
                del banks, state, selector
                torch.cuda.empty_cache()
        assert (counts['answers'], counts['replayed'], counts['visual_calls']) == (180, 72, 1716)
        assert counts['generated_tokens'] <= 23040
        verify()
        (destination / 'completed.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    except BaseException as exc:
        (destination / 'failed.json').write_text(json.dumps(dict(error=repr(exc), **counts), indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
