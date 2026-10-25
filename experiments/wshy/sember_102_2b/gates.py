"""Fixed run identity and resource gates for the S-EMBER diagnostic."""

from __future__ import annotations

import math
from pathlib import Path

from ..online_memory_2b.isolation import output_path, write_root
from ..online_memory_2b.protocol import read, sha, verify_model
from ..online_memory_2b.receipts import inherited_result, successful_counts
from .inputs import FIXED, MANIFEST_SHA, base_inputs, counts, bundle_path, load_bundle, source_lock
from .readout import PROMPT_VERSION


PROFILE = dict(schema='sember-102-baselines-2b-v2', prompt_version=PROMPT_VERSION, **FIXED,
    model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
    precision='float32', canvas_hw=[192, 256], input_fps=1,
    source_pair_encoding='one forward per complete pair shared by five visual banks',
    query_padding=False, language_state='fresh for every answer',
    CPU_pretrained_loads=0, GPU_pretrained_loads=1,
    CPUs=4, RAM_GiB=24, slurm_minutes=45, worker_seconds=2550,
    minimum_free_GPU_GiB=16, minimum_free_RAM_GiB=12,
    maximum_GPU_allocated_GiB=24, maximum_RSS_GiB=20,
    maximum_log_bytes=64 * 2**20, automatic_retry=False,
    content_review='pending', stable_accuracy_claim=False, question_suffix_tokens=512, historical_question_suffix_tokens=128)

SUITES = {
    'online_memory_2b/tests': 28,
    'online_memory_2b/tests_a06': 16,
    'online_memory_2b/tests_ls': 25,
    'long_stream_audit/tests': 8,
    'long_video_2b/tests': 26,
    'sember_102_2b/tests': 11,
}


def verify_prepared(run, *, verify_media=True):
    run = output_path(run)
    protocol = read(run / 'protocol.json')
    if (any(protocol.get(k) != v for k, v in PROFILE.items())
            or protocol.get('run_path') != str(run)
            or protocol.get('write_root') != str(write_root())
            or protocol.get('bundle_path') != str(bundle_path())
            or protocol.get('bundle_manifest_sha256') != MANIFEST_SHA
            or protocol.get('sampling_manifest_sha256') != sha(bundle_path() / 'sampling/manifest.json')):
        raise ValueError('prepared 102-question protocol drift')
    if (sha(bundle_path() / 'sha256.json') != MANIFEST_SHA
            or protocol.get('sampling_manifest_sha256') != sha(bundle_path() / 'sampling/manifest.json')
            or sha(run / 'source_lock.json') != protocol['source_lock_sha256']
            or read(run / 'source_lock.json') != source_lock()):
        raise ValueError('prepared source/input identity drift')
    model = Path(protocol['model_path'])
    if output_path(model) != model or sha(model / 'DOWNLOAD_MANIFEST.json') != protocol['model_manifest_sha256']:
        raise ValueError('model path or manifest drift')
    verify_model(model)
    sources = load_bundle(verify_media=verify_media)
    if protocol.get('sampling_counts') != counts(sources):
        raise ValueError('prepared sampling denominator changed')
    return protocol, sources


def verify_cpu(run, *, verify_media=True):
    protocol, sources = verify_prepared(run, verify_media=verify_media)
    run = output_path(run)
    proof = read(run / 'cpu' / 'preflight.json')
    if (proof.get('status') != 'passed'
            or proof.get('pretrained_loads') != 0
            or proof.get('protocol_sha256') != sha(run / 'protocol.json')
            or proof.get('source_lock_sha256') != sha(run / 'source_lock.json')
            or proof.get('real_RGB_PTS_sources') != FIXED['sources']
            or proof.get('sampled_frames') != counts(sources)['sampled_frames']
            or proof.get('pair_callbacks') != counts(sources)['encoded_pairs']
            or proof.get('queries_checked') != FIXED['requests']
            or proof.get('tiny_answers') != 12
            or proof.get('declared_question_suffix_limit') != 512
            or not 0 < proof.get('maximum_question_suffix_tokens', 0) <= 512
            or proof.get('actual_visual_model_forwards') != 0
            or proof.get('question_token_checks_sha256') != sha(run / 'cpu' / 'question_token_checks.json')
            or proof.get('actual_query_checks_sha256') != sha(run / 'cpu' / 'actual_query_checks.json')):
        raise ValueError('matching successful CPU proof required')
    if (proof.get('suites', {}).keys() != SUITES.keys()
            or proof.get('inherited_log_sha256') != sha(run / 'cpu' / 'inherited.txt')
            or proof.get('inherited') != inherited_result((run / 'cpu' / 'inherited.txt').read_text(encoding='utf-8'))):
        raise ValueError('inherited CPU regression proof changed')
    for folder, expected_count in SUITES.items():
        entry = proof['suites'][folder]
        name = folder.replace('/', '_') + '.txt'
        log = run / 'cpu' / name
        if entry != dict(tests=expected_count, file=name, sha256=sha(log)):
            raise ValueError('CPU contract log identity changed')
        successful_counts(log.read_text(encoding='utf-8'), [expected_count])
    observed = read(run / 'cpu' / 'actual_query_checks.json')
    expected = {(q['id'], item['source']['video_id'])
                for item in sources for q in item['queries']}
    if (len(observed) != FIXED['requests'] or
            {(row['id'], row['source_video_id']) for row in observed} != expected):
        raise ValueError('CPU query receipt coverage changed')
    by_query = {qid: prefix for item in sources for qid, prefix in item['expected_prefix'].items()}
    for row in observed:
        if row['scheduler_prefix'] != by_query[row['id']]:
            raise ValueError('CPU actual query prefix changed')
    tokens = read(run / 'cpu' / 'question_token_checks.json')
    if (len(tokens) != 102 or {row['id'] for row in tokens} != set(by_query)
            or any(type(row['tokens']) is not int or not 0 < row['tokens'] <= 512
                   or row.get('truncated') is not False for row in tokens)
            or max(row['tokens'] for row in tokens) != proof['maximum_question_suffix_tokens']):
        raise ValueError('complete nontruncated tokenized question coverage required')
    return protocol, sources, proof


class Budget:
    def __init__(self):
        self.answers = 0
        self.tokens = 0
        self.seen = set()
        self.expected = {(query['id'], arm) for query in base_inputs()[0] for arm in FIXED['arms']}

    def answer(self, result):
        ids = result.get('generated_ids')
        if (not isinstance(ids, list) or not 1 <= len(ids) <= FIXED['max_new_tokens']
                or any(type(value) is not int or value < 0 for value in ids)
                or not isinstance(result.get('text'), str)):
            raise ValueError('invalid answer or per-answer token budget')
        if self.answers + 1 > FIXED['answers'] or self.tokens + len(ids) > FIXED['max_generated_tokens']:
            raise RuntimeError('fixed answer/token budget exceeded')
        key = (result.get('id'), result.get('arm'))
        if key not in self.expected or key in self.seen:
            raise ValueError('duplicate or invalid answer identity')
        self.seen.add(key)
        self.answers += 1
        self.tokens += len(ids)

    @staticmethod
    def resources(seconds, rss_bytes, gpu_allocated_bytes, log_bytes):
        if (not math.isfinite(seconds) or not 0 <= seconds <= PROFILE['worker_seconds']
                or not 0 <= rss_bytes <= PROFILE['maximum_RSS_GiB'] * 2**30
                or not 0 <= gpu_allocated_bytes <= PROFILE['maximum_GPU_allocated_GiB'] * 2**30
                or not 0 <= log_bytes <= PROFILE['maximum_log_bytes']):
            raise RuntimeError('worker time or resource budget exceeded')
