"""One model and one method per process, causal continuous encoding."""
import argparse
import collections
import gc
import os
from pathlib import Path
import time
from cost_common import ARMS, PREP, ROOT, V0Pairs, read, scale, sha, sync, timed_generate, write


def run_source(env, model, builder, question, arm, guard):
    bank = collections.deque(maxlen=8)
    selector = scale.s.CoverageScorer()
    index = 0
    timing = collections.Counter()
    def commit(first, second, padding):
        nonlocal index
        guard()
        sync(env)
        start = time.perf_counter()
        batch, _ = builder.pair(first, second, 0, 0, synthetic_padding=padding)
        batch = batch.to(model.get_input_embeddings().weight.device)
        sync(env)
        prepared = time.perf_counter()
        features, ids, _ = scale.encode(model, batch, arm, None, None, selector)
        sync(env)
        encoded = time.perf_counter()
        bank.append((index, first.timestamp, second.timestamp, features, ids))
        index += 1
        sync(env)
        finished = time.perf_counter()
        timing['preprocess_transfer_seconds'] += prepared-start
        timing['vision_seconds'] += encoded-prepared
        timing['bank_update_seconds'] += finished-encoded
        timing['callback_seconds'] += finished-start
    sync(env)
    start = time.perf_counter()
    driver = V0Pairs(ROOT / question['media_path'], commit, config='auto4')
    try:
        state = driver.advance(question['query_time_s'])
        sync(env)
        observed = time.perf_counter()
        assert len(bank) == min(index, 8) and index > 0
        assert all(record[2] <= question['query_time_s'] + 1e-6 for record in bank)
        result = timed_generate(env, model, builder, list(bank), question, arm)
    finally:
        driver.close()
    sync(env)
    finished = time.perf_counter()
    result.update(id=question['id'], source_id=question['source_id'], type=question['type'], arm=arm,
                  source_state=state, visible_pairs=[record[0] for record in bank], vision_calls=index,
                  pipeline_wall_seconds=finished-start, observation_seconds=observed-start,
                  decode_sampling_driver_seconds=observed-start-timing['callback_seconds'],
                  encoding=dict(timing), bank_feature_bytes=sum(r[3].numel()*r[3].element_size() for r in bank),
                  bank_index_bytes=sum(r[4].numel()*r[4].element_size() for r in bank),
                  selector_bytes=selector.resident_bytes())
    bank.clear()
    selector.reset()
    return result


def verify():
    lock = read(PREP/'execution_lock.json')
    for name, digest in lock.items():
        path = ROOT/name
        if name.startswith('repo/') and not path.exists():
            path = ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(path) == digest, name
    for side in ['local', 'remote']:
        assert read(PREP/f'CPU_{side}.json')['status'] == 'passed_random_weights_only'
    assert read(PREP/'media_verified.json')['status'] == 'passed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    verify()
    args.out.mkdir(exist_ok=False)
    started = time.monotonic()
    def guard():
        if time.monotonic()-started > args.seconds:
            raise TimeoutError('Approved job deadline')
    job = read(PREP/'jobs.json')[args.job]
    references = {(r['id'], r['arm']): r for r in read(PREP/'reference.json')}
    questions = {q['id']: q for q in read(PREP/'panel.json')}
    count = 0
    try:
        env = scale.s.runtime()
        torch = env['torch']
        torch.set_num_threads(1)
        assert torch.cuda.device_count() == 1 and env['psutil'].virtual_memory().available >= 12*2**30
        tick = time.monotonic()
        model, info = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            ROOT/'models/Qwen3.5-2B', local_files_only=True, dtype=torch.float32, device_map='cuda',
            attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
        model.eval().requires_grad_(False)
        versions = {name: param._version for name, param in model.named_parameters()}
        builder = scale.s.make_builder(env, env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B', local_files_only=True))
        sync(env)
        load_seconds = time.monotonic()-tick
        with torch.no_grad(), env['fp32_reference']():
            warm = run_source(env, model, builder, read(PREP/'warmup.json'), job['arm'], guard)
            write(args.out/'startup.json', dict(model_loads=1, load_seconds=load_seconds, warmup=warm))
            del warm
            gc.collect()
            torch.cuda.empty_cache()
            with (args.out/'answers.jsonl').open('x', encoding='utf-8') as stream:
                import json
                for question_id in job['question_ids']:
                    guard()
                    torch.cuda.reset_peak_memory_stats()
                    result = run_source(env, model, builder, questions[question_id], job['arm'], guard)
                    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                  host_RSS_end_bytes=env['psutil'].Process().memory_info().rss)
                    reference = references[(question_id, job['arm'])]
                    compared = ['generated_ids', 'input_ids_sha256', 'position_ids_sha256', 'visual_tokens', 'visible_pairs', 'source_state']
                    result['reference_match'] = {name: result[name] == reference[name] for name in compared}
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                    stream.flush()
                    assert all(result['reference_match'].values()), ('frozen endpoint mismatch', question_id, result['reference_match'])
                    count += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(json.dumps(dict(job=args.job, completed=count, seconds=time.monotonic()-started)), flush=True)
        assert all(param._version == versions[name] and param.grad is None for name, param in model.named_parameters())
        write(args.out/'completed.json', dict(status='complete', job=job, questions=count, seconds=time.monotonic()-started, base_unchanged=True))
    except Exception as error:
        write(args.out/'failed.json', dict(error=repr(error), completed_questions=count, seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
