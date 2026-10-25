"""Original-model full forward versus the unchanged external-feature layout."""
import hashlib
import importlib.util
import sys

from ..online_memory_2b.frozen import REPO
from .plan import PROFILE


def original_module():
    # Import under an explicit private package, so active qwen35_local remains
    # the streaming implementation. Relative imports inside the original work.
    name = '_sember_original_qwen35'
    if name not in sys.modules:
        folder = REPO / 'qwen35_local'
        spec = importlib.util.spec_from_file_location(name, folder / '__init__.py',
                                                      submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_inputs(env, model, builder, packets, question, metadata):
    """Same audited original-model adapter as fixed_stage2, without truncation.

    The original processor emits a marker for each temporal group. The pinned
    model expects one grid per marker for positional indexing. Expand ONLY the
    indexing grid, keep the actual video/vision grid intact. This compatibility
    operation is explicit in receipts; never claim this is unmodified generate.
    """
    if not packets or len(packets) % 2:
        raise ValueError('complete original pairs required')
    torch = env['torch']
    meta = dict(metadata, frames_indices=[p.frame_id for p in packets])
    native = builder.processor(text=[builder.template(question)],
        videos=[env['np'].stack([p.rgb for p in packets])], video_metadata=[meta],
        do_resize=False, do_sample_frames=False, return_tensors='pt')
    if native.input_ids.shape[1] + 8 > PROFILE['context_tokens']:
        raise ValueError('native context exceeds fixed ceiling; no truncation')
    grid = torch.repeat_interleave(native.video_grid_thw, native.video_grid_thw[:, 0], 0).clone()
    grid[:, 0] = 1
    coords, _ = model.model.get_rope_index(**dict(native, video_grid_thw=grid))
    native['position_ids'] = torch.cat((torch.arange(native.input_ids.shape[1])[None, None, :], coords), 0)
    return native


def check_same_pixels(env, builder, packets, native):
    tensors = []
    for i in range(0, len(packets), 2):
        pair, _ = builder.pair(packets[i], packets[i+1], 0, 0, synthetic_padding=False)
        if pair.video_grid_thw.tolist() != [[1, 12, 16]]:
            raise ValueError('unexpected pair grid')
        tensors.append(pair.pixel_values_videos)
    joined = env['torch'].cat(tensors)
    if not env['torch'].equal(native.pixel_values_videos, joined):
        raise ValueError('native/current preprocessed pixels differ')
    return tensor_sha(joined)


def native_answer(env, model, builder, inputs):
    """One complete original-model prefill; own cache released after each QA."""
    torch = env['torch']
    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)
    saved = model.model.rope_deltas
    result = cache = None
    try:
        result = model(**inputs, use_cache=True, logits_to_keep=1)
        logits, cache = result.logits, result.past_key_values
        first_sha = tensor_sha(logits[0, -1].float())
        n = inputs.input_ids.shape[1]
        rope = int(inputs.position_ids[1:].max()) + 1
        configured = model.generation_config.eos_token_id
        stops = set(configured if isinstance(configured, list) else [configured])
        stops.add(builder.processor.tokenizer.eos_token_id)
        ids, reason = [], 'budget'
        for _ in range(8):
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite native logits')
            token = int(logits[0, -1].argmax())
            ids.append(token)
            if token in stops:
                reason = 'eos'
                break
            result = model(input_ids=torch.tensor([[token]], device=device),
                position_ids=torch.tensor([[[n]], [[rope]], [[rope]], [[rope]]], device=device),
                cache_position=torch.tensor([n], device=device),
                attention_mask=torch.ones((1, n+1), dtype=torch.long, device=device),
                past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits, cache = result.logits, result.past_key_values
            n += 1
            rope += 1
        return dict(text=builder.processor.tokenizer.decode(ids, skip_special_tokens=True),
            generated_ids=ids, stop_reason=reason, input_ids_sha256=tensor_sha(inputs.input_ids),
            position_ids_sha256=tensor_sha(inputs.position_ids), first_logits_sha256=first_sha,
            input_tokens=inputs.input_ids.shape[1], visual_tokens=int(
                (inputs.input_ids == builder.processor.video_token_id).sum()))
    finally:
        del result, cache
        model.model.rope_deltas = saved


def current_answer(env, model, builder, nodes, question):
    """Existing layout/positions and 512-token prefill; larger explicit ceiling
    only to fit the complete evidence. No node merging or dropped old pairs.
    """
    torch = env['torch']
    if any(n.depth != 0 or n.metadata()['original_pairs'] != 1 for n in nodes):
        raise ValueError('compressed node in uncompressed control')
    if nodes:
        emb, pos, ids, meta = env['context'](model, builder, nodes, question,
            limits=env['Limits'](PROFILE['context_tokens'], 14400, 300), max_new_tokens=8)
    else:
        emb, pos, ids, meta = env['pilot'].text_context(env, model, builder, question)
    if emb.shape[1] + 8 > PROFILE['context_tokens']:
        raise ValueError('current context exceeds fixed ceiling; no truncation')
    session = env['HybridSequence'](model, max_tokens=PROFILE['context_tokens'], session_id='evidence-diagnostic')
    try:
        for i in range(0, emb.shape[1], 512):
            end = min(i+512, emb.shape[1])
            logits = session.append(inputs_embeds=emb[:, i:end], position_ids=pos[:, :, i:end],
                                    logits_mode='last' if end == emb.shape[1] else 'none')
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite current logits')
        first_sha = tensor_sha(logits[0, -1].float())
        configured = model.generation_config.eos_token_id
        stops = set(configured if isinstance(configured, list) else [configured])
        stops.add(builder.processor.tokenizer.eos_token_id)
        generated, reason = env['greedy_decode'](session, logits, 8, stops)
        return dict(text=builder.processor.tokenizer.decode(generated, skip_special_tokens=True),
            generated_ids=generated, stop_reason=reason, input_ids_sha256=tensor_sha(ids),
            position_ids_sha256=tensor_sha(pos), first_logits_sha256=first_sha,
            input_tokens=meta['input_tokens'], visual_tokens=meta['visual_tokens'])
    finally:
        session.cache.reset()


def isolated(env, model, function):
    torch = env['torch']
    rope = model.model.rope_deltas
    rope_copy = rope.clone() if isinstance(rope, torch.Tensor) else None
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    try:
        return function()
    finally:
        if (model.model.rope_deltas is not rope
                or (rope_copy is not None and not torch.equal(rope, rope_copy))
                or not torch.equal(cpu, torch.get_rng_state())
                or (cuda and any(not torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all())))):
            raise RuntimeError('QA changed persistent position/RNG state')
