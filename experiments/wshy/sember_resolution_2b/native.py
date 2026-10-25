"""Resolution-only input ceiling and bounded native attention workspace.

The original full forward, all K/V and its language cache remain intact.
Only the query dimension of full-attention SDPA is tiled; no prefill streaming,
frame removal, feature compression, alternate dtype or model-weight change.
"""
import contextlib
import json
import os
import sys
import threading
import time

CONTEXT_NATIVE = 40960
QUERY_TILE = 512
WORKER_SECONDS = 21300


def inputs(env, model, builder, packets, prompt, metadata):
    torch = env['torch']
    if not packets:
        rendered = builder.processor.apply_chat_template([dict(role='user',content=prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        native = builder.processor.tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
        n = native.input_ids.shape[1]
        native['position_ids'] = torch.arange(n)[None,None,:].expand(4,1,-1)
    else:
        if len(packets)%2:
            raise ValueError('unpaired real frame')
        meta = dict(metadata,frames_indices=[p.frame_id for p in packets])
        native = builder.processor(text=[builder.template(prompt)],
            videos=[env['np'].stack([p.rgb for p in packets])], video_metadata=[meta],
            do_resize=False, do_sample_frames=False, return_tensors='pt')
        h,w = packets[0].rgb.shape[:2]
        if native.video_grid_thw.tolist() != [[len(packets)//2,h//16,w//16]]:
            raise ValueError('processor changed declared grid')
        visual = int((native.input_ids==builder.processor.video_token_id).sum())
        if visual != len(packets)//2*(h//32)*(w//32):
            raise ValueError('unexpected visual token count')
        grid = torch.repeat_interleave(native.video_grid_thw,native.video_grid_thw[:,0],0).clone()
        grid[:,0] = 1
        coords,_ = model.model.get_rope_index(**dict(native,video_grid_thw=grid))
        native['position_ids'] = torch.cat((torch.arange(native.input_ids.shape[1])[None,None,:],coords),0)
        timestamps = builder.processor._calculate_timestamps(meta['frames_indices'],meta['fps'],2)
        if (len(timestamps)!=len(packets)//2 or max(abs(t-(a.timestamp+b.timestamp)/2)
                for t,a,b in zip(timestamps,packets[::2],packets[1::2])) > 1/meta['fps']+1e-6):
            raise ValueError('processor timestamps do not match real PTS')
    if native.input_ids.shape[1]+8 > CONTEXT_NATIVE:
        raise ValueError('40K context exceeded; no truncation or fallback')
    return native


def tiled_attention(original, module, query, key, value, attention_mask,
                    dropout=0.0, scaling=None, is_causal=None, tile=QUERY_TILE, **kwargs):
    import torch
    length, keys = query.shape[-2], key.shape[-2]
    if tile < 1 or dropout != 0.0 or module.training:
        raise ValueError('positive tile and evaluation-only zero-dropout attention required')
    if length <= tile:
        return original(module, query, key, value, attention_mask, dropout=dropout,
                        scaling=scaling, is_causal=is_causal, **kwargs)
    if keys < length:
        raise ValueError('native causal self-attention requires all keys')
    causal = getattr(module, 'is_causal', True) if is_causal is None else is_causal
    output = None
    for start in range(0, length, tile):
        end = min(start+tile, length)
        if attention_mask is None:
            # Match upstream SDPA: implicit causal masking uses upper-left
            # alignment. Cached non-square prefills must supply their mask.
            if causal and keys != length:
                raise ValueError('cached multi-query attention requires an explicit mask')
            mask = (torch.arange(keys, device=query.device)[None, :] <=
                    torch.arange(start, end, device=query.device)[:, None]) if causal else None
        else:
            if attention_mask.shape[-2] not in (1, length):
                raise ValueError('attention mask query dimension mismatch')
            mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[..., start:end, :]
        part, weights = original(module, query[..., start:end, :], key, value, mask,
                                 dropout=0.0, scaling=scaling, is_causal=False, **kwargs)
        if weights is not None:
            raise ValueError('SDPA attention weights unexpectedly materialized')
        if output is None:
            output = part.new_empty((part.shape[0], length, part.shape[2], part.shape[3]))
        output[:, start:end] = part
        del part, mask
    return output, None


@contextlib.contextmanager
def query_tiling(model, tile=QUERY_TILE):
    """Route only this native model's full language attention; restore on error."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    registry = ALL_ATTENTION_FUNCTIONS
    original = registry['sdpa']
    targets = {id(m) for m in model.modules() if m.__class__.__name__ == 'Qwen3_5Attention'}
    if not targets:
        raise ValueError('no native full-attention layers found')
    # This job has one model/forward thread at a time. Vision and all unrelated
    # modules retain the exact upstream implementation.
    def dispatch(module, query, key, value, attention_mask, **kwargs):
        if id(module) in targets:
            return tiled_attention(original, module, query, key, value, attention_mask,
                                   tile=tile, **kwargs)
        return original(module, query, key, value, attention_mask, **kwargs)
    registry['sdpa'] = dispatch
    try:
        yield
    finally:
        registry['sdpa'] = original


def bounded_answer(env, model, builder, native, tile=QUERY_TILE):
    from ..sember_readout_diagnostic.inference import native_answer
    with query_tiling(model, tile):
        return native_answer(env, model, builder, native)


def watchdog(env, out):
    """Six-hour proposal: stop at 5h55m or 40 GiB process-tree RSS."""
    deadline = float(os.environ['QWEN_DIAGNOSTIC_STARTED']) + WORKER_SECONDS
    def check():
        while True:
            try:
                proc = env['psutil'].Process()
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except env['psutil'].NoSuchProcess:
                        pass
                if time.time() > deadline or rss > 40*2**30:
                    with (out/'resource_failure.json').open('x') as stream:
                        json.dump(dict(status='failed',reason='wall-time or process-tree RSS cap',
                                       rss_bytes=rss,time=time.time()),stream)
                    os._exit(124)
            except Exception as exc:
                print('resource monitor failure: '+repr(exc), file=sys.stderr, flush=True)
                os._exit(125)
            time.sleep(1)
    threading.Thread(target=check, daemon=True).start()

