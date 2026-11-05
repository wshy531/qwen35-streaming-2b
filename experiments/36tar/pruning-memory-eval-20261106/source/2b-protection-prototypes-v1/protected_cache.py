"""Experimental read-time hybrid splice, NOT FlashBack reproduction.

Build history and recent separately with identical shared prompt prefix. Recent
RoPE must already use the intended joint coordinates (text cache index is local).
At question time concatenate attention K/V, retain recent conv/recurrent state.
This protects recent PREFILL, not necessarily the subsequent answer from distraction.
"""
import torch

@torch.inference_mode()
def compose(history,recent,shared_prefix):
    if history.model is not recent.model:raise ValueError('same frozen model required')
    if not 0<shared_prefix<=min(history.seen_tokens,recent.seen_tokens):raise ValueError('shared prefix length')
    length=history.seen_tokens+recent.seen_tokens-shared_prefix
    if length>recent.max_tokens:raise ValueError('combined budget exceeded')
    out=recent.fork('protected-readout')
    for field in ('key_cache','value_cache'):
        hs=getattr(history.cache,field);rs=getattr(recent.cache,field);dst=getattr(out.cache,field)
        if len(hs)!=len(rs):raise ValueError('layer mismatch')
        for i,(a,b) in enumerate(zip(hs,rs)):
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError('cache topology mismatch')
                continue
            if a.shape[-2]!=history.seen_tokens or b.shape[-2]!=recent.seen_tokens:raise ValueError('cache length mismatch')
            if not torch.equal(a[...,:shared_prefix,:],b[...,:shared_prefix,:]):raise ValueError('shared prompt KV differs')
            dst[i]=torch.cat((a,b[...,shared_prefix:,:]),dim=-2).clone()
    out.seen_tokens=length
    out.next_rope_position=max(history.next_rope_position,recent.next_rope_position)
    assert out.cache.get_seq_length()==length
    return out
