"""Shared causal decode, visual write and isolated QA with a unique module name."""
from dataclasses import dataclass
import gc
import hashlib
import sys
import time
import numpy as np
import torch
from common import ROOT, ARMS
from audit_media import canvas
from interference_readout import compose
from sember_banks import distance
from sember_nodes import digest
from sember_common import THRESHOLD


@dataclass(frozen=True)
class Packet:
    frame_id: int
    timestamp: float
    rgb: object


def exact_pairs(path, trace):
    """Retain only one pending sampled RGB frame; verify the frozen causal trace."""
    import av
    if not trace or len(trace) % 2 or any(a['frame_id'] >= b['frame_id'] or a['pts_s'] >= b['pts_s']
                                          for a, b in zip(trace, trace[1:])):
        raise ValueError('ordered complete-pair trace required')
    cursor, pending, origin = 0, None, None
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        video.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video)):
            if origin is None:
                origin = float(frame.time)
            target = trace[cursor]
            if index < target['frame_id']:
                continue
            pts = float(frame.time) - origin
            image = canvas(frame)
            if (index != target['frame_id'] or abs(pts - target['pts_s']) > 1e-7
                    or hashlib.sha256(image.tobytes()).hexdigest() != target['mid_rgb_sha256']):
                raise ValueError('actual sampled pixels or PTS differ from frozen input')
            packet = Packet(index, pts, np.asarray(image).copy())
            if pending is None:
                pending = packet
            else:
                yield pending, packet
                pending = None
            cursor += 1
            if cursor == len(trace):
                break
    if cursor != len(trace) or pending is not None:
        raise ValueError('incomplete decoded causal prefix')


@torch.inference_mode()
def project(env, model, builder, first, second):
    batch, record = builder.pair(first, second, 0, 0)
    batch = batch.to(model.get_input_embeddings().weight.device)
    saved = model.model.rope_deltas
    try:
        with env['fp32_reference']():
            vision = model.model.get_video_features(batch.pixel_values_videos, batch.video_grid_thw, return_dict=True)
            features = torch.cat(vision.pooler_output, 0).detach().to(device='cpu', dtype=torch.float32)
        if features.shape != (108, model.get_input_embeddings().embedding_dim) or not torch.isfinite(features).all():
            raise ValueError('complete finite 288x384 projected pair required')
        return features, record
    finally:
        model.model.rope_deltas = saved


@torch.inference_mode()
def append(bank, value, index, first, second, previous_descriptor):
    from sember_nodes import pair_node
    descriptor = value.mean(0)
    starts = previous_descriptor is None or distance(previous_descriptor, descriptor) > THRESHOLD
    bank.append(pair_node(value, index, first.timestamp, second.timestamp, (9, 12)), starts_event=starts)
    return descriptor.clone()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def ask(env, model, builder, bank, question, arm, donors, *, max_new_tokens=8):
    if arm not in ARMS or not 1 <= max_new_tokens <= 8:
        raise ValueError('fixed arm and bounded generation required')
    before = digest(bank.nodes())
    saved = model.model.rope_deltas
    sequence = None
    device = model.get_input_embeddings().weight.device
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    sync()
    started = time.monotonic()
    try:
        with env['fp32_reference']():
            emb, pos, ids, meta = compose(model, builder, bank, question, arm, donors)
            sequence = env['HybridSequence'](model, max_tokens=8192, session_id='history-interference-' + arm)
            logits = None
            for begin in range(0, emb.shape[1], 512):
                end = min(begin + 512, emb.shape[1])
                logits = sequence.append(inputs_embeds=emb[:, begin:end], position_ids=pos[:, :, begin:end],
                    logits_mode='last' if end == emb.shape[1] else 'none')
            if logits is None or not torch.isfinite(logits).all():
                raise FloatingPointError('invalid QA logits')
            sync()
            prefilled = time.monotonic()
            eos = model.generation_config.eos_token_id
            stops = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
            stops.add(builder.processor.tokenizer.eos_token_id)
            tokens, reason = env['greedy_decode'](sequence, logits, max_new_tokens, stops)
            sync()
            finished = time.monotonic()
            if not 1 <= len(tokens) <= max_new_tokens:
                raise ValueError('generated count outside declared budget')
            return dict(text=builder.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                generated_ids=tokens, stop_reason=reason, input_ids_sha256=tensor_sha(ids),
                position_ids_sha256=tensor_sha(pos), QA_seconds=finished-started,
                prefill_and_context_seconds=prefilled-started, decode_seconds=finished-prefilled,
                branch_released=True, **meta)
    finally:
        if sequence is not None:
            sequence.cache.reset()
        model.model.rope_deltas = saved
        if digest(bank.nodes()) != before:
            raise RuntimeError('QA changed visual memory')
