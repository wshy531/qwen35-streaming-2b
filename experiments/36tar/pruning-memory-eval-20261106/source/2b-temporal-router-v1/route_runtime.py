"""Text-only fresh hybrid prefill; no visual encoding and no token generation."""
from route_common import ROOT
import sys
import time
from collections.abc import Mapping


def runtime():
    sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
    from sember_common import runtime as load
    return load()


def tokenized(tokenizer,prompt):
    messages=[dict(role='system',content=prompt['system']),dict(role='user',content=prompt['user'])]
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=False,return_tensors='pt')
    ids=encoded['input_ids'] if isinstance(encoded,Mapping) else encoded
    if ids.ndim!=2 or ids.shape[0]!=1 or not 1<=ids.shape[1]<=1024:
        raise ValueError('complete bounded text-only router prompt required; no truncation')
    codes=[tokenizer.encode(c,add_special_tokens=False) for c in prompt['codes']]
    if any(len(v)!=1 for v in codes) or len({v[0] for v in codes})!=len(codes):
        raise ValueError('distinct single-token category codes required')
    return ids,[v[0] for v in codes]


def score_view(env,model,tokenizer,prompt):
    torch=env['torch']
    ids,codes=tokenized(tokenizer,prompt)
    device=model.get_input_embeddings().weight.device
    sequence=None;saved=model.model.rope_deltas
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    sync();started=time.monotonic()
    try:
        with torch.inference_mode(),env['fp32_reference']():
            sequence=env['HybridSequence'](model,max_tokens=1024,session_id='temporal-router-query-local')
            logits=None
            for begin in range(0,ids.shape[1],512):
                end=min(begin+512,ids.shape[1])
                logits=sequence.append(input_ids=ids[:,begin:end].to(device),
                    logits_mode='last' if end==ids.shape[1] else 'none')
            values=logits[0,-1,codes].float().detach().cpu()
            if not torch.isfinite(values).all():raise FloatingPointError('nonfinite classifier logits')
            sync()
            import hashlib
            return dict(candidate_logits=values.tolist(),input_tokens=ids.shape[1],
                input_ids_sha256=hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
                seconds=time.monotonic()-started,branch_released=True,
                generated_tokens=0,visual_forwards=0)
    finally:
        if sequence is not None:sequence.cache.reset()
        model.model.rope_deltas=saved
