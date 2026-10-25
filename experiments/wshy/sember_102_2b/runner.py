"""Prepare and CPU-check the 102-question S-EMBER baseline without real model weights."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ..online_memory_2b.isolation import configure, output_path, write_root

configure()

from ..online_memory_2b.frozen import REPO, runtime  # noqa: E402
from ..online_memory_2b.protocol import (deployment_identity, read, save, sha,
    verify_model, verify_package)  # noqa: E402
from ..online_memory_2b.receipts import inherited_result, successful_counts  # noqa: E402
from ..online_memory_2b.runner import environment, isolated_readout  # noqa: E402
from ..long_video_2b.scheduler import decode_samples, replay_source  # noqa: E402
from ..long_video_2b.driver import run_sessions  # noqa: E402
from .gates import PROFILE, SUITES, verify_prepared  # noqa: E402
from .inputs import (FIXED, MANIFEST_SHA, counts, bundle_path,
                     check_prefix, load_bundle, source_lock)  # noqa: E402


def prepare(run, model):
    run = output_path(run)
    model = output_path(model)
    if run.exists():
        raise ValueError('new CPU/GPU run directory required')
    sources = load_bundle()
    verify_package()
    verify_model(model)
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'source_lock.json', source_lock())
    save(run / 'protocol.json', dict(**PROFILE, run_path=str(run),
        write_root=str(write_root()), bundle_path=str(bundle_path()), model_path=str(model),
        bundle_manifest_sha256=MANIFEST_SHA, sampling_manifest_sha256=sha(bundle_path() / 'sampling/manifest.json'),
        sampling_counts=counts(sources),
        model_manifest_sha256=sha(model / 'DOWNLOAD_MANIFEST.json'),
        source_lock_sha256=sha(run / 'source_lock.json'),
        deployment=deployment_identity(), created_UTC=datetime.now(timezone.utc).isoformat()))


def cpu(run):
    protocol, sources = verify_prepared(run)
    run = output_path(run)
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('SLURM_CPUS_PER_TASK') != '4'
            or int(os.environ.get('SLURM_RESTART_COUNT', '0')) != 0):
        raise ValueError('approved CPU Slurm allocation, hidden GPU and no restart required')
    out = output_path(run / 'cpu')
    out.mkdir(exist_ok=False)

    def execute(command, name):
        with (out / name).open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=REPO, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            raise RuntimeError('CPU validation failed: ' + name)
        return (out / name).read_text(encoding='utf-8')

    inherited_log = execute([sys.executable, '-B', str(REPO / 'scripts/run_cpu_checks.py'),
                             '--model-dir', protocol['model_path']], 'inherited.txt')
    inherited = inherited_result(inherited_log)
    suites = {}
    for folder, count in SUITES.items():
        name = folder.replace('/', '_') + '.txt'
        output = execute([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                          str(REPO / 'experiments/wshy' / folder), '-v'], name)
        successful_counts(output, [count])
        suites[folder] = dict(tests=count, file=name, sha256=sha(out / name))
    if read(run / 'source_lock.json') != source_lock():
        raise ValueError('source changed during CPU contracts')

    from .readout import complete_question_runtime, SUFFIX_TOKENS
    env = complete_question_runtime(runtime())
    torch = env['torch']
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError('GPU must remain hidden for CPU preflight')
    observed = []
    counters = []
    for item in sources:
        frames, media = item['frames'], item['media']
        if sha(media) != item['source']['source_sha256']:
            raise ValueError('media changed before CPU decode')
        def check(query, prefix):
            check_prefix(item['expected_prefix'][query['id']], prefix)
            observed.append(dict(id=query['id'], source_video_id=item['source']['video_id'],
                                 scheduler_prefix=prefix))
        result = replay_source(frames, item['queries'], decode_samples(env, media, frames),
                               lambda *unused: None, check)
        if sha(media) != item['source']['source_sha256']:
            raise ValueError('media changed after CPU decode')
        counters.append(result)
    if (sum(c['sampled_frames'] for c in counters) != counts(sources)['sampled_frames']
            or sum(c['encoded_pairs'] for c in counters) != counts(sources)['encoded_pairs']
            or len(observed) != FIXED['requests']):
        raise ValueError('actual RGB/PTS or question coverage denominator')
    processor = env['AutoProcessor'].from_pretrained(protocol['model_path'], local_files_only=True)
    builder = env['PairInputBuilder'](processor, canvas=(192, 256), resize_mode='letterbox')
    all_queries = [q for item in sources for q in item['queries']]
    suffix_checks = [dict(id=q['id'], tokens=builder.question_ids(env['prompt'](q)).shape[1],
        prompt_sha256=hashlib.sha256(env['prompt'](q).encode()).hexdigest(), truncated=False)
        for q in all_queries]
    maximum = max(row['tokens'] for row in suffix_checks)
    if maximum > SUFFIX_TOKENS:
        raise ValueError('full question/options exceed declared 512-token allowance; no truncation')
    torch.manual_seed(123)
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **env['tiny_config'](processor), attn_implementation='sdpa')).eval()
    tiny_frames = [dict(frame_id=i, pts_s=float(i), rgb_sha256='0' * 64) for i in range(24)]
    tiny_queries = [dict(id=f'tiny:{i}', source_path='synthetic.mp4', query_time_s=t,
                         question='Return only A.', options=['first', 'second'], cohort='synthetic')
                    for i, t in enumerate((17, 23))]
    longest_id = max(suffix_checks, key=lambda row: row['tokens'])['id']
    longest = next(q for q in all_queries if q['id'] == longest_id)
    tiny_queries[-1].update(question=longest['question'], options=list(longest['options']))
    answers = []

    def encode(index, first, second):
        tensor = torch.randn(48, 32, generator=torch.Generator().manual_seed(800 + index))
        return env['legacy'].features.raw_block(tensor, index, first.timestamp, second.timestamp)

    packets = (SimpleNamespace(frame_id=f['frame_id'], timestamp=f['pts_s']) for f in tiny_frames)
    with torch.inference_mode(), env['fp32_reference']():
        tiny = run_sessions(tiny_frames, tiny_queries, packets, encode,
            lambda nodes, question: isolated_readout(env, model, builder, nodes, question),
            env['prompt'], lambda event: None,
            lambda row: answers.append(dict(id=row['id'], arm=row['arm'],
                                            generated=len(row['generated_ids']))))
    if tiny['answers'] != 12 or len(answers) != 12:
        raise RuntimeError('twelve tiny-model answers required')
    save(out / 'actual_query_checks.json', observed)
    save(out / 'question_token_checks.json', suffix_checks)
    save(out / 'preflight.json', dict(status='passed', pretrained_loads=0,
        protocol_sha256=sha(run / 'protocol.json'), source_lock_sha256=sha(run / 'source_lock.json'),
        environment=environment(env), inherited=inherited,
        inherited_log_sha256=sha(out / 'inherited.txt'), suites=suites,
        real_RGB_PTS_sources=FIXED['sources'], sampled_frames=counts(sources)['sampled_frames'],
        pair_callbacks=counts(sources)['encoded_pairs'], queries_checked=FIXED['requests'],
        actual_visual_model_forwards=0, maximum_question_suffix_tokens=maximum, declared_question_suffix_limit=SUFFIX_TOKENS,
        actual_query_checks_sha256=sha(out / 'actual_query_checks.json'),
        question_token_checks_sha256=sha(out / 'question_token_checks.json'),
        tiny_longest_official_question_id=longest_id,
        tiny_model=tiny, tiny_answers=len(answers), tiny_answer_records=answers,
        model_accuracy_evaluated=False, independent_semantic_review='pending'))
