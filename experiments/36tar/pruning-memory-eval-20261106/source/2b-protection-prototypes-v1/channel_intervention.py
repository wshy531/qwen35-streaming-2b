"""Readout-state ablation, not a deployable or equivalence-preserving model.

The two source caches have identical token length and intended joint coordinates.
Copy full-attention KV from one and GDN convolution/recurrent states from the other.
This intervenes on a bundle of states, not on every causal path independently.
"""
import torch

@torch.inference_mode()
def cross_channels(attention,linear):
    if attention.model is not linear.model:raise ValueError('one frozen model')
    if attention.seen_tokens!=linear.seen_tokens:raise ValueError('same joint context length')
    if attention.next_rope_position!=linear.next_rope_position:raise ValueError('same joint RoPE endpoint')
    out=linear.fork('channel-intervention')
    for field in ('key_cache','value_cache'):
        src=getattr(attention.cache,field);dst=getattr(out.cache,field)
        if len(src)!=len(dst):raise ValueError('cache topology')
        for i,(a,b) in enumerate(zip(src,dst)):
            if (a is None)!=(b is None):raise ValueError('layer topology')
            if a is not None:
                if a.shape!=b.shape:raise ValueError('cache shape')
                dst[i]=a.clone()
    return out
