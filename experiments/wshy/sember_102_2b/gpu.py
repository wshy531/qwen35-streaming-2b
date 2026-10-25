"""One FP32 model load and a fixed 612-answer official-question grid."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time

from ..online_memory_2b.isolation import configure, output_path

configure()

from ..online_memory_2b.frozen import REPO, runtime  # noqa: E402
from ..online_memory_2b.protocol import save, sha, verify_model  # noqa: E402
from ..online_memory_2b.receipts import successful_counts  # noqa: E402
from ..online_memory_2b.runner import environment, isolated_readout  # noqa: E402
from ..long_video_2b.scheduler import decode_samples  # noqa: E402
from ..long_video_2b.driver import run_sessions  # noqa: E402
from .gates import Budget, FIXED, PROFILE, SUITES, verify_cpu  # noqa: E402
from .inputs import check_prefix, counts  # noqa: E402


def gpu(run):
    protocol, sources, proof = verify_cpu(run)
    run = output_path(run)
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1')
            or os.environ.get('SLURM_CPUS_PER_TASK') != '4'
            or int(os.environ.get('SLURM_RESTART_COUNT', '0')) != 0):
        raise ValueError('approved one-GPU allocation, four CPUs and no restart required')
    verify_model(protocol['model_path'])
    out = output_path(run / 'gpu')
    out.mkdir(exist_ok=False)
    # Run new contracts with GPU hidden before loading weights.
    log = out / 'baseline_contracts.txt'
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
            str(REPO / 'experiments/wshy/sember_102_2b/tests'), '-v'],
            cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
            stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    if result.returncode:
        raise RuntimeError('baseline contracts failed before pretrained load')
    successful_counts(log.read_text(encoding='utf-8'), [SUITES['sember_102_2b/tests']])
    verify_cpu(run)
    from .readout import complete_question_runtime
    env = complete_question_runtime(runtime())
    torch = env['torch']
    if torch.cuda.device_count() != 1 or environment(env) != proof['environment']:
        raise ValueError('GPU allocation or CPU/GPU software identity mismatch')
    torch.set_num_threads(1)
    torch.manual_seed(123)
    free, _ = torch.cuda.mem_get_info()
    if (free < PROFILE['minimum_free_GPU_GiB'] * 2**30
            or env['psutil'].virtual_memory().available < PROFILE['minimum_free_RAM_GiB'] * 2**30):
        raise MemoryError('preload GPU/RAM resource gate')
    save(out / 'guard.json', dict(job=os.environ['SLURM_JOB_ID'], GPU=torch.cuda.get_device_name(),
        free_GPU_bytes=free, protocol_sha256=sha(run / 'protocol.json'),
        CPU_preflight_sha256=sha(run / 'cpu/preflight.json')))
    save(out / 'load_started.json', dict(maximum_pretrained_loads=1, automatic_retry=False))
    started = time.monotonic()
    model = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
        protocol['model_path'], local_files_only=True, dtype=torch.float32,
        device_map='cuda', attn_implementation='sdpa').eval()
    if (model.config.text_config.hidden_size != 2048
            or any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError('FP32 Qwen3.5-2B model identity required')
    save(out / 'loaded.json', dict(pretrained_loads=1, seconds=time.monotonic() - started))
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        protocol['model_path'], local_files_only=True), canvas=(192, 256), resize_mode='letterbox')
    budget = Budget()
    vision_calls = encodes = peak_rss = peak_bank = 0
    timings = dict(decode=0., encode=0., answer=0.)
    summaries = []

    def guard():
        nonlocal peak_rss
        import resource
        peak_rss = max(peak_rss, env['psutil'].Process().memory_info().rss,
                       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        log_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        Budget.resources(time.monotonic() - started, peak_rss,
                         torch.cuda.max_memory_allocated(), log_bytes)

    def hook(*unused):
        nonlocal vision_calls
        vision_calls += 1

    handle = model.model.visual.register_forward_hook(hook)
    try:
        with (out / 'predictions.jsonl').open('x', encoding='utf-8') as predictions, \
             (out / 'events.jsonl').open('x', encoding='utf-8') as events, \
             torch.inference_mode(), env['fp32_reference']():
            def write(stream, row):
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()

            def readout(nodes, question):
                guard()
                torch.cuda.synchronize()
                tick = time.monotonic()
                result = isolated_readout(env, model, builder, nodes, question)
                torch.cuda.synchronize()
                timings['answer'] += time.monotonic() - tick
                if vision_calls != encodes:
                    raise RuntimeError('question unexpectedly encoded visual input')
                return result

            for item in sources:
                guard()
                source, media, frames = item['source'], item['media'], item['frames']
                if sha(media) != source['source_sha256']:
                    raise ValueError('media changed before GPU decode')
                bank_bytes = {}

                def encode(index, first, second):
                    nonlocal encodes
                    guard()
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    inputs, _ = builder.pair(first, second, 0, 0, synthetic_padding=False)
                    features = torch.cat(model.model.get_video_features(
                        inputs.pixel_values_videos.to('cuda'),
                        inputs.video_grid_thw.to('cuda'), return_dict=True).pooler_output, 0)
                    block = env['legacy'].features.raw_block(features, index,
                                                             first.timestamp, second.timestamp)
                    del inputs, features
                    torch.cuda.synchronize()
                    timings['encode'] += time.monotonic() - tick
                    encodes += 1
                    if vision_calls != encodes or encodes > counts(sources)['encoded_pairs']:
                        raise RuntimeError('repeated, missing or excess visual encoding')
                    return block

                def event(row):
                    nonlocal peak_bank
                    if row['event'] == 'append':
                        state = row['state']
                        if (not state['valid'] or state['closed'] or state['visual_tokens'] > 384
                                or len(state['nodes']) > 8 or state['resident_bytes'] > 8*394560+8192
                                or state['metadata_bytes'] > 32768 or state['full_history_retained']
                                or state['language_cache_retained']):
                            raise RuntimeError('bounded visual memory or isolation violated')
                        bank_bytes[row['arm']] = state['resident_bytes']
                    elif row['event'] == 'query':
                        expected = item['expected_prefix'][row['id']]
                        check_prefix(expected, {k: row[k] for k in expected})
                    else:
                        raise ValueError('unexpected driver event')
                    write(events, dict(source_path=source['source_path'], **row))
                    if row['event'] == 'append' and row['arm'] == 'reservoir4':
                        combined = sum(bank_bytes.values())
                        peak_bank = max(peak_bank, combined)
                        write(events, dict(event='resources', source_path=source['source_path'],
                            pair=row['pair'], combined_bank_bytes=combined,
                            GPU_allocated=torch.cuda.memory_allocated(),
                            GPU_reserved=torch.cuda.memory_reserved(),
                            RSS_bytes=env['psutil'].Process().memory_info().rss))
                    guard()

                def answer(row):
                    check_prefix(item['expected_prefix'][row['id']], row['prefix'])
                    budget.answer(row)
                    write(predictions, row)
                    guard()

                def samples():
                    iterator = decode_samples(env, media, frames)
                    try:
                        while True:
                            tick = time.monotonic()
                            try:
                                packet = next(iterator)
                            except StopIteration:
                                return
                            timings['decode'] += time.monotonic() - tick
                            yield packet
                            del packet
                    finally:
                        iterator.close()

                summary = run_sessions(frames, item['queries'], samples(), encode, readout,
                                       env['prompt'], event, answer)
                summaries.append(dict(source_path=source['source_path'], **summary))
                if sha(media) != source['source_sha256']:
                    raise ValueError('media changed during GPU run')
                gc.collect()
                guard()
                print(json.dumps(dict(sources=len(summaries), answers=budget.answers,
                    generated_tokens=budget.tokens, visual_forwards=vision_calls)), flush=True)
            if (budget.answers != FIXED['answers'] or budget.seen != budget.expected
                    or encodes != counts(sources)['encoded_pairs']
                    or vision_calls != counts(sources)['encoded_pairs']
                    or sum(s['sampled_frames'] for s in summaries) != counts(sources)['sampled_frames']
                    or sum(s['reset_checks'] for s in summaries) != FIXED['reset_checks']):
                raise RuntimeError('complete 102-question/18-source baseline coverage required')
            guard()
            save(out / 'completed.json', dict(status='complete', pretrained_loads=1,
                main_answers=budget.answers, generated_tokens=budget.tokens,
                source_pairs_encoded=encodes, visual_forward_calls=vision_calls,
                reset_checks=FIXED['reset_checks'], all_QA_banks_unchanged=True,
                sources=summaries, peak_combined_bank_bytes=peak_bank,
                peak_RSS_bytes=peak_rss, peak_GPU_allocated=torch.cuda.max_memory_allocated(),
                peak_GPU_reserved=torch.cuda.max_memory_reserved(),
                seconds=time.monotonic() - started, stage_seconds=timings,
                environment=environment(env), protocol_sha256=sha(run / 'protocol.json'),
                CPU_preflight_sha256=sha(run / 'cpu/preflight.json'),
                baseline_contract_log_sha256=sha(log),
                output_sha256={name: sha(out / name) for name in ('predictions.jsonl', 'events.jsonl')},
                independent_semantic_review='pending', stable_accuracy_claim=False))
    finally:
        handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
