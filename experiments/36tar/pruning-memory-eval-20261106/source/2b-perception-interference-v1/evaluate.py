"""One separately approved GPU allocation; 90 formal plus 3 warm-up answers."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
import time
from common import ROOT, PREP, INPUTS, ARMS, read, save, sha, displayed, require_development_review
sys.path.insert(0, str(ROOT / 'experiments/2b-sember-v1'))


def main(out, lock_path, media_root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '4':
        raise RuntimeError('explicitly approved one-GPU/four-CPU Slurm allocation required')
    if not out.resolve().is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('owned result directory required')
    from freeze_execution import verify_lock
    lock = verify_lock(lock_path)
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    review = read(PREP / 'assistant_review_LOCAL_ONLY.json')
    require_development_review(manifest, review)
    from sember_common import runtime
    from sember_banks import Bank
    from sember_runtime import make_builder
    from sember_protocol import prompt
    from interference_execution import exact_pairs, project, append, ask
    env = runtime()
    torch, psutil = env['torch'], env['psutil']
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 24 * 2**30:
        raise RuntimeError('one GPU with declared free-memory margin required')
    if psutil.virtual_memory().available < 12 * 2**30:
        raise RuntimeError('insufficient system memory margin')
    media = {r['pair_id']: r for r in read(PREP / 'cluster_media_CPU.json')['sources']}
    for pair in manifest['pairs']:
        path = media_root / Path(pair['source_path']).name
        if path.resolve().parent != media_root.resolve() or path.stat().st_size != pair['media_bytes'] or sha(path) != pair['media_sha256']:
            raise ValueError('frozen exact media mismatch')
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    calls = [0]
    def guard():
        if time.monotonic() - started > 1050 or psutil.Process().memory_info().rss > 20 * 2**30:
            raise RuntimeError('fixed worker time/RSS limit exceeded')
        if torch.cuda.max_memory_allocated() > 48 * 2**30:
            raise RuntimeError('fixed GPU allocation-memory limit exceeded')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > 32 * 2**20:
            raise RuntimeError('fixed result limit exceeded')
    model, loading = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
        ROOT / 'models/Qwen3.5-2B', local_files_only=True, dtype=torch.float32,
        device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError('pretrained loading mismatch')
    model.eval()
    if model.config.text_config.hidden_size != 2048:
        raise ValueError('frozen Qwen3.5-2B projection width required')
    builder = make_builder(env, env['AutoProcessor'].from_pretrained(ROOT / 'models/Qwen3.5-2B', local_files_only=True))
    def visual_hook(module, args):
        calls[0] += 1
    hook = model.model.visual.register_forward_pre_hook(visual_hook)
    save(out / 'startup.json', dict(pretrained_loads=1, precision='float32',
        loading_summary={k: len(loading.get(k) or ()) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')},
        execution_lock_sha256=sha(lock_path), load_seconds=time.monotonic()-started,
        review_scope=review['scope'], trained=False))
    tasks = {t['id']: t for t in read(INPUTS / 'tasks_LOCAL_ONLY.json')}
    schedule = read(INPUTS / 'schedule.json')
    formal = generated = warm_tokens = warm_answers = 0
    profiles = []
    with (out / 'answers.jsonl').open('x', encoding='utf-8') as stream:
        for case_index, pair in enumerate(manifest['pairs']):
            guard()
            pid = pair['pair_id']
            donor_case = next(p for p in manifest['pairs'] if p['source_path'] == pair['donor_source_path'])
            donor_path = media_root / Path(donor_case['source_path']).name
            donors = []
            for first, second in exact_pairs(donor_path, media[donor_case['pair_id']]['donor_trace']):
                guard()
                value, _ = project(env, model, builder, first, second)
                donors.append(value)
                del value
            if len(donors) != 4:
                raise ValueError('four donor pairs required')
            bank = Bank('LS', 108)
            descriptor = None
            pair_count = 0
            target_path = media_root / Path(pair['source_path']).name
            for first, second in exact_pairs(target_path, media[pid]['sampled_trace']):
                guard()
                value, _ = project(env, model, builder, first, second)
                descriptor = append(bank, value, pair_count, first, second, descriptor)
                pair_count += 1
                del value
            if pair_count != pair['committed_pairs'] or bank.last_end != pair['committed_through_s']:
                raise ValueError('query prefix drifted')
            del first, second
            state = bank.status()
            if state['tokens'] > 864 or state['recent_tokens'] != 432:
                raise RuntimeError('bounded bank violation')
            if case_index == 0:
                warm = []
                text = prompt(displayed(tasks[pid + ':current'], 0)[0])
                for arm in ARMS:
                    result = ask(env, model, builder, bank, text, arm, donors)
                    warm.append(result)
                    warm_tokens += len(result['generated_ids'])
                    warm_answers += 1
                save(out / 'warmup.json', warm)
                warm_ids = {r['arm']: r['generated_ids'] for r in warm}
                del warm
                gc.collect()
                torch.cuda.empty_cache()
            for item in [r for r in schedule if r['pair_id'] == pid]:
                guard()
                task = tasks[item['task_id']]
                question, mapping = displayed(task, item['shift'])
                result = ask(env, model, builder, bank, prompt(question), item['arm'], donors)
                if case_index == 0 and task['role'] == 'current' and item['shift'] == 0:
                    if result['generated_ids'] != warm_ids[item['arm']]:
                        raise RuntimeError('same-input warm/formal QA changed after isolated reads')
                record = dict(task_id=task['id'], pair_id=pid, role=task['role'], shift=item['shift'],
                    display_to_original=mapping, query_time_s=pair['query_time_s'],
                    committed_through_s=pair['committed_through_s'], **result)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                formal += 1
                generated += len(result['generated_ids'])
            profiles.append(dict(pair_id=pid, target_complete_pairs=pair_count,
                donor_source_pair_id=donor_case['pair_id'], donor_pairs=4, bank=state,
                donor_tensor_bytes=sum(v.numel()*v.element_size() for v in donors),
                no_full_feature_archive=True, no_observation_language_state=True))
            del donors, bank, descriptor
            gc.collect()
            torch.cuda.empty_cache()
            print(dict(completed_cases=case_index+1, formal_answers=formal, visual_calls=calls[0]), flush=True)
    guard()
    hook.remove()
    if formal != 90 or warm_answers != 3 or calls[0] != 367 or generated + warm_tokens > 744:
        raise RuntimeError('actual work differed from frozen resource budget')
    save(out / 'completed.json', dict(status='complete', formal_answers=formal, warmup_answers=warm_answers,
        generated_tokens=generated, warmup_generated_tokens=warm_tokens, visual_calls=calls[0],
        pretrained_loads=1, profiles=profiles, wall_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        RSS_bytes=psutil.Process().memory_info().rss, trained=False, automatic_retry=False,
        causal_file_replay_not_wallclock_1x=True, review_scope=review['scope']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--media-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.out, args.lock, args.media_root)
