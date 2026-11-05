"""Independent continuous cost check using the frozen explanation protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'runs/stream_prune_cost_prepare_20261102_01'
spec = importlib.util.spec_from_file_location('scale_protocol', ROOT / 'experiments/2b-evidence-scale-v1/common.py')
scale = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scale)
sys.path.insert(0, str(ROOT / 'experiments/streamingbench-router-v1'))
from v0_source import V0Pairs

ARMS = ('full', 'fixed12', 'heuristic11')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def sync(env):
    if env['torch'].cuda.is_available():
        env['torch'].cuda.synchronize()


def timed_generate(env, model, builder, records, question, arm, limit=128):
    """Same operations as scale.generate, with synchronized phase boundaries."""
    sync(env)
    started = time.perf_counter()
    embeds, positions, ids = scale.layout(env, model, builder, records, question, arm)
    assert ids.shape[1] + limit <= 4096
    sync(env)
    built = time.perf_counter()
    branch = env['HybridSequence'](model, max_tokens=4096, session_id='stream-prune-cost')
    saved = model.model.rope_deltas
    try:
        for start in range(0, embeds.shape[1], 512):
            end = min(start + 512, embeds.shape[1])
            logits = branch.append(inputs_embeds=embeds[:, start:end], position_ids=positions[:, :, start:end],
                                   logits_mode='last' if end == embeds.shape[1] else 'none')
        sync(env)
        prefilled = time.perf_counter()
        eos = model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        stops.add(builder.processor.tokenizer.eos_token_id)
        generated, reason = env['greedy_decode'](branch, logits, limit, stops)
        sync(env)
        decoded = time.perf_counter()
    finally:
        branch.cache.reset()
        model.model.rope_deltas = saved
    text = builder.processor.tokenizer.decode(generated, skip_special_tokens=True)
    result = dict(text=text, generated_ids=generated, reason=reason, input_tokens=ids.shape[1],
                  input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                  position_ids_sha256=hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),
                  visual_tokens=sum(len(r[3]) for r in records), **scale.parse(text, len(question['options']), 'B'))
    sync(env)
    result['timing'] = dict(context_seconds=built-started, prefill_seconds=prefilled-built,
                            generation_seconds=decoded-prefilled, QA_total_seconds=time.perf_counter()-started,
                            prefill_latency_scope='Until next-token logits ready; excludes token selection and delivery')
    return result
