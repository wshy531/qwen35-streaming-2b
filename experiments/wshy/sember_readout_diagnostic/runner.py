"""Local plan, approved CPU preflight, two sequential model loads, offline score."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..online_memory_2b.isolation import configure, output_path, write_root
configure()
from ..online_memory_2b.frozen import REPO, runtime
from ..online_memory_2b.protocol import read, save, sha, verify_model, verify_package
from ..sember_102_2b.inputs import base_inputs, bundle_path, load_bundle, source_lock as base_lock
from ..sember_102_2b.readout import complete_question_runtime, question_prompt
from .plan import ARMS, PROFILE, digest, requests, select
from .inference import (original_module, native_inputs, native_answer, current_answer,
                        check_same_pixels, isolated, tensor_sha)

SAMPLING_SHA = '72b3d3189b2f6a6731fb057db81c647061d1a1323f980503ca7b9af29b956a1e'
OLD_PREDICTIONS_SHA = 'f24eabee54c74b5c01126e1f7eef9963d0a9f0096e9114775f13c3f89acfbc72'


def code_lock():
    result = base_lock()
    for folder in (Path(__file__).parent, REPO / 'qwen35_local'):
        for p in folder.rglob('*'):
            if p.suffix in ('.py', '.sh'):
                result[p.relative_to(REPO).as_posix()] = sha(p)
    p = REPO / 'scripts/run_2b_sember_readout_diagnostic.py'
    result[p.relative_to(REPO).as_posix()] = sha(p)
    return dict(sorted(result.items()))


def plan(out):
    out = output_path(out)
    queries, _, _ = base_inputs()
    selected = requests(queries, read(bundle_path() / 'labels-offline-only.json'))
    if len(selected) != PROFILE['questions']:
        raise ValueError('all 102 original questions required')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'requests.json', selected)
    save(out / 'source_lock.json', code_lock())
    save(out / 'protocol.json', dict(PROFILE, requests_sha256=sha(out / 'requests.json'),
        source_lock_sha256=sha(out / 'source_lock.json'), sampling_manifest_sha256=SAMPLING_SHA))
    return out


def verify(out):
    out = output_path(out)
    protocol = read(out / 'protocol.json')
    if (any(protocol.get(k) != v for k, v in PROFILE.items())
            or protocol['requests_sha256'] != sha(out / 'requests.json')
            or protocol['source_lock_sha256'] != sha(out / 'source_lock.json')
            or read(out / 'source_lock.json') != code_lock()
            or protocol['sampling_manifest_sha256'] != SAMPLING_SHA):
        raise ValueError('locked diagnostic changed')
    queries, _, _ = base_inputs()
    planned = read(out / 'requests.json')
    # No labels enter inference. Compare exact question/order/prompt provenance.
    if ([r['query'] for r in planned] != queries
            or any(r['prompt_sha256'] != hashlib.sha256(question_prompt(r['query']).encode()).hexdigest()
                   for r in planned)):
        raise ValueError('question identity mismatch')
    return protocol, planned


def allocation(gpu=False):
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '4'
            or int(os.environ.get('SLURM_RESTART_COUNT', '0')) != 0
            or (gpu and os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1'))
            or (not gpu and os.environ.get('CUDA_VISIBLE_DEVICES') != '')):
        raise ValueError('approved four-CPU allocation, correct GPU visibility, no restart required')


def sampling():
    if sha(bundle_path() / 'sampling/manifest.json') != SAMPLING_SHA:
        raise ValueError('original RGB/PTS sampling manifest changed')
    return load_bundle()


def software(env):
    from ..online_memory_2b.runner import environment
    import transformers
    for module in (env['torch'], env['np'], env['av'], env['psutil'], transformers):
        output_path(module.__file__)
    output_path(sys.executable)
    return environment(env)


def source_packets(env, item):
    from ..long_video_2b.scheduler import decode_samples
    if sha(item['media']) != item['source']['source_sha256']:
        raise ValueError('media changed')
    with env['av'].open(str(item['media'])) as container:
        stream = container.streams.video[0]
        if not stream.average_rate or not stream.duration or not stream.frames:
            raise ValueError('native video metadata incomplete')
        metadata = dict(fps=float(stream.average_rate), total_num_frames=stream.frames,
                        duration=float(stream.duration * stream.time_base))
    packets = {p.frame_id: p for p in decode_samples(env, item['media'], item['frames'])}
    return packets, metadata


def get_selection(item, request):
    return select(item['frames'], request['query']['query_time_s'], request['evidence_interval'])


def chosen_packets(packets, selection, kind):
    return [packets[f['frame_id']] for f in selection[kind]]


def nodes_for(env, model, builder, packets, indices, *, random=False):
    nodes = []
    for j, index in enumerate(indices):
        a, b = packets[2*j:2*j+2]
        if random:
            visual = env['torch'].zeros(48, model.config.text_config.hidden_size)
        else:
            pair, _ = builder.pair(a, b, 0, 0, synthetic_padding=False)
            device = model.get_input_embeddings().weight.device
            visual = env['torch'].cat(model.model.get_video_features(
                pair.pixel_values_videos.to(device), pair.video_grid_thw.to(device),
                return_dict=True).pooler_output, 0)
        nodes.append(env['legacy'].forest.raw(env['legacy'].features.raw_block(
            visual, index, a.timestamp, b.timestamp)))
    return nodes


def cpu(out, model_path):
    allocation()
    out, model_path = output_path(out), output_path(model_path)
    verify(out)
    verify_package()
    verify_model(model_path)
    records = verify(out)[1]
    sources = sampling()
    directory = out / 'cpu'
    directory.mkdir(exist_ok=False)
    with (directory / 'contracts.txt').open('x', encoding='utf8') as log:
        for folder in ('sember_102_2b/tests', 'sember_readout_diagnostic/tests'):
            subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                            str(REPO / 'experiments/wshy' / folder), '-v'],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
    env = complete_question_runtime(runtime())
    torch = env['torch']
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise ValueError('CPU preflight must hide GPU')
    identity = software(env)
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        model_path, local_files_only=True), canvas=(192, 256), resize_mode='letterbox')
    original = original_module()
    config = env['tiny_config'](builder.processor)
    torch.manual_seed(123)
    original_model = original.Qwen3_5ForConditionalGeneration(
        original.Qwen3_5Config(**config, attn_implementation='sdpa')).eval()
    current_model = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**config, attn_implementation='sdpa')).eval()
    audited, tiny, selected_manifest = [], [], []
    with torch.inference_mode(), env['fp32_reference']():
        for item in sources:
            packets, metadata = source_packets(env, item)
            for request in records:
                q = request['query']
                if q['source_path'] != item['source']['source_path']:
                    continue
                selection = get_selection(item, request)
                selected_manifest.append(dict(id=q['id'], **selection))
                question = question_prompt(q)
                if builder.question_ids(question).shape[1] > 512:
                    raise ValueError('question exceeds declared allowance')
                check = dict(id=q['id'], prompt_sha256=request['prompt_sha256'],
                             evidence_status=selection['evidence_status'])
                for kind in ('prefix', 'evidence'):
                    chosen = chosen_packets(packets, selection, kind)
                    if not chosen:
                        continue
                    inputs = native_inputs(env, original_model, builder, chosen, question, metadata)
                    pixel_sha = check_same_pixels(env, builder, chosen, inputs)
                    # Processor timestamps derive from real frame IDs and source FPS.
                    rendered = builder.processor._calculate_timestamps(
                        [p.frame_id for p in chosen], metadata['fps'], 2)
                    error = max(abs(t-(a.timestamp+b.timestamp)/2)
                                for t, a, b in zip(rendered, chosen[::2], chosen[1::2]))
                    if len(rendered) != len(chosen)//2 or error > 1/metadata['fps'] + 1e-6:
                        raise ValueError('native timestamp reconstruction differs from actual PTS')
                    check[kind] = dict(input_tokens=inputs.input_ids.shape[1],
                        pixels_sha256=pixel_sha, timestamp_max_error_s=error,
                        native_position_ids_sha256=tensor_sha(inputs.position_ids))
                evidence = chosen_packets(packets, selection, 'evidence')
                if evidence:
                    nodes = nodes_for(env, current_model, builder, evidence,
                                      selection['evidence_pair_indices'], random=True)
                    _, _, _, meta = env['context'](current_model, builder, nodes, question,
                        limits=env['Limits'](PROFILE['context_tokens'], 14400, 300), max_new_tokens=8)
                    check['current_input_tokens'] = meta['input_tokens']
                    del nodes
                if not tiny:
                    chosen = chosen_packets(packets, selection, 'prefix')[:4]
                    for count in (2, 4):
                        subset = chosen[:count]
                        inputs = native_inputs(env, original_model, builder, subset, question, metadata)
                        native = isolated(env, original_model, lambda: native_answer(
                            env, original_model, builder, inputs))
                        nodes = nodes_for(env, current_model, builder, subset,
                                          list(range(len(subset)//2)), random=True)
                        current = isolated(env, current_model, lambda: current_answer(
                            env, current_model, builder, nodes, question))
                        tiny.extend([dict(arm='native', generated=len(native['generated_ids'])),
                                     dict(arm='current', generated=len(current['generated_ids']))])
                        del nodes, inputs
                audited.append(check)
            del packets
    if len(audited) != 102 or len(tiny) != 4:
        raise ValueError('CPU coverage incomplete')
    verify(out)
    save(directory / 'selections.json', selected_manifest)
    save(directory / 'inputs.json', audited)
    save(directory / 'passed.json', dict(status='passed', pretrained_loads=0,
        questions=102, sources=18, tiny_answers=tiny, environment=identity,
        protocol_sha256=sha(out / 'protocol.json'), selections_sha256=sha(directory / 'selections.json'),
        inputs_sha256=sha(directory / 'inputs.json'), contracts_sha256=sha(directory / 'contracts.txt'),
        model_path=str(model_path), model_manifest_sha256=sha(model_path / 'DOWNLOAD_MANIFEST.json')))


def gpu(out):
    allocation(gpu=True)
    out = output_path(out)
    protocol, records = verify(out)
    proof = read(out / 'cpu/passed.json')
    if (proof['status'] != 'passed' or proof['pretrained_loads'] != 0
            or proof['protocol_sha256'] != sha(out / 'protocol.json')
            or any(proof[field] != sha(out / 'cpu' / name) for field, name in (
                ('selections_sha256', 'selections.json'), ('inputs_sha256', 'inputs.json'),
                ('contracts_sha256', 'contracts.txt')))):
        raise ValueError('matching CPU proof required')
    model_path = output_path(proof['model_path'])
    verify_model(model_path)
    if proof['model_manifest_sha256'] != sha(model_path / 'DOWNLOAD_MANIFEST.json'):
        raise ValueError('model manifest changed')
    sources = sampling()
    checks = {r['id']: r for r in read(out / 'cpu/inputs.json')}
    selections = {r['id']: {k: v for k, v in r.items() if k != 'id'}
                  for r in read(out / 'cpu/selections.json')}
    directory = out / 'gpu'
    directory.mkdir(exist_ok=False)
    env = complete_question_runtime(runtime())
    torch = env['torch']
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or software(env) != proof['environment']:
        raise ValueError('GPU count or CPU/GPU environment mismatch')
    started = time.monotonic()
    deadline = float(os.environ['QWEN_DIAGNOSTIC_STARTED']) + PROFILE['worker_seconds']
    count = generated = loads = 0
    seen = set()
    torch.manual_seed(123)
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        model_path, local_files_only=True), canvas=(192, 256), resize_mode='letterbox')

    def guard():
        import resource
        if time.time() > deadline:
            raise TimeoutError('approved job deadline; do not resubmit')
        peak_rss = max(env['psutil'].Process().memory_info().rss,
                       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if (peak_rss > PROFILE['maximum_RSS_GiB']*2**30
                or torch.cuda.max_memory_allocated() > PROFILE['maximum_GPU_allocated_GiB']*2**30):
            raise MemoryError('diagnostic resource ceiling')

    def emit(stream, q, arm, **result):
        nonlocal count, generated
        key = (q['id'], arm)
        ids = result.get('generated_ids', [])
        if key in seen or len(ids) > 8 or count >= 408 or generated + len(ids) > 3264:
            raise ValueError('answer/token budget or duplicate')
        seen.add(key)
        count += 1
        generated += len(ids)
        stream.write(json.dumps(dict(id=q['id'], arm=arm, source_path=q['source_path'],
            query_time_s=q['query_time_s'], **result), ensure_ascii=False)+'\n')
        stream.flush()

    with (directory / 'predictions.jsonl').open('x', encoding='utf8') as stream:
        for original in (True, False):
            guard()
            verify(out)
            if torch.cuda.mem_get_info()[0] < PROFILE['minimum_free_GPU_GiB']*2**30:
                raise MemoryError('requires >=64 GiB free GPU memory; no smaller-input fallback')
            cls = original_module().Qwen3_5ForConditionalGeneration if original else env['Qwen3_5ForConditionalGeneration']
            tag = 'original' if original else 'current'
            save(directory / (tag + '_load_started.json'), dict(attempt=loads+1, class_name=cls.__module__))
            model = cls.from_pretrained(model_path, local_files_only=True, dtype=torch.float32,
                                        device_map='cuda', attn_implementation='sdpa').eval()
            loads += 1
            if (model.config.text_config.hidden_size != 2048
                    or any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters())):
                raise ValueError('FP32 2B identity required')
            calls = [0]
            def hook(*unused):
                calls[0] += 1
            handle = model.model.visual.register_forward_hook(hook)
            try:
                with torch.inference_mode(), env['fp32_reference']():
                    for item in sources:
                        guard()
                        packets, metadata = source_packets(env, item)
                        for request in records:
                            q = request['query']
                            if q['source_path'] != item['source']['source_path']:
                                continue
                            selection = get_selection(item, request)
                            if selection != selections[q['id']]:
                                raise ValueError('CPU/GPU selected frame identity differs')
                            question = question_prompt(q)
                            arms = ('native_prefix', 'native_evidence') if original else ('current_evidence', 'text_only_clean')
                            for arm in arms:
                                guard()
                                before = calls[0]
                                kind = 'prefix' if arm == 'native_prefix' else 'evidence'
                                chosen = chosen_packets(packets, selection, kind)
                                if arm != 'text_only_clean' and not chosen:
                                    emit(stream, q, arm, status='no_committed_evidence_sample', text='')
                                    continue
                                if original:
                                    inputs = native_inputs(env, model, builder, chosen, question, metadata)
                                    if (tensor_sha(inputs.pixel_values_videos) != checks[q['id']][kind]['pixels_sha256']
                                            or tensor_sha(inputs.position_ids) != checks[q['id']][kind]['native_position_ids_sha256']):
                                        raise ValueError('CPU/GPU native input drift')
                                    result = isolated(env, model, lambda: native_answer(env, model, builder, inputs))
                                    del inputs
                                    expected_calls = 1
                                else:
                                    if arm == 'text_only_clean':
                                        nodes = []
                                    else:
                                        # Recheck processed pixels against the CPU proof before encoding.
                                        pair_pixels = [builder.pair(chosen[i], chosen[i+1], 0, 0,
                                            synthetic_padding=False)[0].pixel_values_videos
                                            for i in range(0, len(chosen), 2)]
                                        if tensor_sha(torch.cat(pair_pixels)) != checks[q['id']]['evidence']['pixels_sha256']:
                                            raise ValueError('CPU/GPU current pixel drift')
                                        del pair_pixels
                                        nodes = nodes_for(env, model, builder, chosen, selection['evidence_pair_indices'])
                                    bank = [(n.metadata(), tensor_sha(n.features), tensor_sha(n.mass), tensor_sha(n.coords)) for n in nodes]
                                    result = isolated(env, model, lambda: current_answer(env, model, builder, nodes, question))
                                    if bank != [(n.metadata(), tensor_sha(n.features), tensor_sha(n.mass), tensor_sha(n.coords)) for n in nodes]:
                                        raise RuntimeError('readout changed evidence features')
                                    expected_calls = len(nodes)
                                    result['node_metadata'] = [n.metadata() for n in nodes]
                                    del nodes
                                if calls[0] - before != expected_calls:
                                    raise RuntimeError('unexpected visual encoding count')
                                emit(stream, q, arm, status='ok', prompt_sha256=request['prompt_sha256'],
                                    frames_sha256=None if arm == 'text_only_clean' else selection[kind+'_sha256'],
                                    visual_forward_calls=expected_calls, **result)
                                guard()
                        del packets
                        print(json.dumps(dict(model=tag, answers=count, generated_tokens=generated)), flush=True)
            finally:
                handle.remove()
                del model
                gc.collect()
                torch.cuda.empty_cache()
    if seen != {(r['query']['id'], arm) for r in records for arm in ARMS} or loads != 2:
        raise ValueError('incomplete diagnostic grid')
    verify(out)
    save(directory / 'completed.json', dict(status='complete', records=count, generated_tokens=generated,
        pretrained_loads=loads, seconds=time.monotonic()-started,
        peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        predictions_sha256=sha(directory / 'predictions.jsonl'),
        CPU_proof_sha256=sha(out / 'cpu/passed.json'), protocol_sha256=sha(out / 'protocol.json')))


def score(out):
    from .scoring import build
    out = output_path(out)
    verify(out)
    complete = read(out / 'gpu/completed.json')
    if (complete['status'] != 'complete'
            or complete['predictions_sha256'] != sha(out / 'gpu/predictions.jsonl')
            or complete['protocol_sha256'] != sha(out / 'protocol.json')
            or complete['CPU_proof_sha256'] != sha(out / 'cpu/passed.json')):
        raise ValueError('completed output integrity failed')
    predictions = [json.loads(line) for line in (out / 'gpu/predictions.jsonl').read_text(encoding='utf8').splitlines()]
    old = write_root() / 'runs/sember-102-eval-20261024-01/gpu/predictions.jsonl'
    if sha(old) != OLD_PREDICTIONS_SHA:
        raise ValueError('historical predictions changed')
    historical = [json.loads(line) for line in old.read_text(encoding='utf8').splitlines()]
    queries, _, _ = base_inputs()
    summary, detail = build(queries, read(bundle_path() / 'labels-offline-only.json'), predictions, historical)
    directory = out / 'score'
    directory.mkdir(exist_ok=False)
    save(directory / 'summary.json', summary)
    save(directory / 'details.json', detail)
