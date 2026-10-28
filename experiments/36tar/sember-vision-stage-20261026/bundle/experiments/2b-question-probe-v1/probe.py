"""Question-aware raw-feature selection; no split-depth persistent cache.

Inspired by FastV attention ranking, not its layerwise hidden-state pruning.
Only single-sample unpadded FP32 eval inference is supported. A private stop
exception exits immediately at the first full-attention module's entry.
"""
from dataclasses import replace
import importlib
import time
import torch


class _ProbeDone(Exception):
    pass


def _rotate(x, cos, sin, rotate_half):
    width = cos.shape[-1]
    part, rest = x[..., :width], x[..., width:]
    return torch.cat((part*cos[:, None] + rotate_half(part)*sin[:, None], rest), -1)


@torch.inference_mode()
def attention_rows(attn, hidden, position_embeddings, query_rows, *, row_chunk=16):
    """Exact causal softmax rows, normalized over ALL eligible keys.

    The full N x N attention matrix is never formed. Q/K projection includes
    Qwen's output-gate split, RMS normalization, partial RoPE and grouped KV.
    The caller must provide one unpadded, fresh prefix with no external mask.
    """
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.dtype != torch.float32:
        raise ValueError('single FP32 unpadded sample required')
    if type(row_chunk) is not int or not 1 <= row_chunk <= 16:
        raise ValueError('bounded score rows required')
    rows = torch.as_tensor(query_rows, dtype=torch.long, device=hidden.device)
    n = hidden.shape[1]
    if (rows.ndim != 1 or not 0 < len(rows) <= 128 or rows.min() < 0 or rows.max() >= n
            or not torch.equal(rows, rows.unique(sorted=True)) or n > 2048):
        raise ValueError('sorted, bounded nonempty question rows required')
    mod = importlib.import_module(type(attn).__module__)
    cos, sin = position_embeddings
    key = attn.k_norm(attn.k_proj(hidden).view(1, n, -1, attn.head_dim)).transpose(1, 2)
    key = _rotate(key, cos, sin, mod.rotate_half)
    key = mod.repeat_kv(key, attn.num_key_value_groups)
    summed = torch.zeros(n, device=hidden.device, dtype=torch.float32)
    peak = 0
    for indices in rows.split(row_chunk):
        count = len(indices)
        q, _ = torch.chunk(attn.q_proj(hidden[:, indices]).view(1, count, -1, attn.head_dim*2), 2, -1)
        q = attn.q_norm(q.reshape(1, count, -1, attn.head_dim)).transpose(1, 2)
        q = _rotate(q, cos[:, indices], sin[:, indices], mod.rotate_half)
        logits = (q @ key.transpose(-1, -2))*attn.scaling
        future = torch.arange(n, device=hidden.device)[None, :] > indices[:, None]
        logits.masked_fill_(future[None, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, -1, dtype=torch.float32)
        summed += weights[0].sum((0, 1))
        # Both the logits and probabilities can be live during softmax.
        peak = max(peak, 2*weights.numel()*weights.element_size())
    return summed/(len(rows)*attn.config.num_attention_heads), dict(
        query_rows=len(rows), key_tokens=n, score_row_chunk=row_chunk,
        score_matrix_live_bytes_bound=peak, full_N_by_N_attention_materialized=False)


def question_rows(builder, question, input_tokens):
    """Use the exact text span, including supplied MCQ options, excluding chat controls."""
    suffix = builder.question_ids(question)[0]
    rendered = builder.template(question).split(builder.marker)[1]
    encoded = builder.processor.tokenizer(rendered, add_special_tokens=False,
                                          return_offsets_mapping=True)
    if encoded['input_ids'] != suffix.tolist():
        raise ValueError('offset tokenizer must match frozen question IDs')
    start = rendered.find(question)
    if start < 0 or rendered.find(question, start+1) != -1:
        raise ValueError('unique verbatim question span required')
    end = start+len(question)
    selected = [input_tokens-len(suffix)+i for i, (a, b) in enumerate(encoded['offset_mapping'])
                if a < end and b > start and b > a]
    if not selected or len(selected) > 128:
        raise ValueError('question text exceeds declared probe budget')
    return selected


@torch.inference_mode()
def score_visual(model, builder, question, emb, pos, ids, *, row_chunk=16):
    if model.training or emb.shape[1] != ids.shape[1] or pos.shape != (4, 1, ids.shape[1]):
        raise ValueError('matching eval context required')
    text = model.model.language_model
    if list(text.config.layer_types[:4]) != ['linear_attention']*3+['full_attention']:
        raise ValueError('registered three-GDN then full-attention topology required')
    if getattr(text, '_qprobe_active', False):
        raise RuntimeError('concurrent/nested probe unsupported')
    rows = question_rows(builder, question, ids.shape[1])
    target = text.layers[3].self_attn
    scores = stats = None
    saved_rope = model.model.rope_deltas
    text._qprobe_active = True
    handle = None
    device = emb.device
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    sync(); began = time.monotonic()

    def stop(module, args, kwargs):
        nonlocal scores, stats
        if kwargs.get('past_key_values') is not None:
            raise ValueError('probe must never own persistent hybrid cache')
        scores, stats = attention_rows(module, kwargs['hidden_states'],
                                       kwargs['position_embeddings'], rows, row_chunk=row_chunk)
        raise _ProbeDone()

    try:
        handle = target.register_forward_pre_hook(stop, with_kwargs=True)
        try:
            text(inputs_embeds=emb, position_ids=pos, use_cache=False,
                 attention_mask=torch.ones(ids.shape, dtype=torch.long, device=device),
                 past_key_values=None, output_attentions=False, output_hidden_states=False)
        except _ProbeDone:
            pass
        else:
            raise RuntimeError('probe failed to stop before attention output')
        if scores is None:
            raise RuntimeError('probe produced no scores')
        sync()
        visual = (ids[0].to(device) == builder.processor.video_token_id)
        return scores[visual].detach().cpu(), dict(**stats, probe_seconds=time.monotonic()-began,
            completed_decoder_layers=3, scored_attention_layer_one_based=4,
            probe_cache_created=False, probe_logits_computed=False,
            query_text_tokens_only=True, correct_answer_labels_used=False)
    finally:
        if handle is not None:
            handle.remove()
        text._qprobe_active = False
        model.model.rope_deltas = saved_rope


def select_half(nodes, scores, *, full=False):
    """Fixed per-node half budget; stable ties; original temporal/spatial order."""
    if scores.ndim != 1 or len(scores) != sum(len(n.features) for n in nodes) or not torch.isfinite(scores).all():
        raise ValueError('one finite score per stored visual token required')
    result, selected, start = [], [], 0
    for node in nodes:
        count = len(node.features)
        if count < 2 or count % 2:
            raise ValueError('registered even nodes required')
        keep = count if full else count//2
        index = torch.argsort(scores[start:start+count], descending=True, stable=True)[:keep].sort().values
        result.append(replace(node, features=node.features[index].clone(),
                              mass=node.mass[index].clone(), coords=node.coords[index].clone()))
        selected.append(index.tolist());start += count
    return result, selected
