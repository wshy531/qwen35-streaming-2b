"""Original hybrid decoder in an empty context; vision access is forbidden."""
import time
import torch
from text_common import emit_decode,cache_bytes


@torch.inference_mode()
def answer(env,model,processor,ids,callback,limit=16):
    if model.training or next(model.parameters()).dtype!=torch.float32 or not 1<=limit<=16:raise ValueError('fixed FP32 eval')
    device=model.get_input_embeddings().weight.device
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]+limit>1024:raise ValueError('complete bounded text input required')
    seq=env['HybridSequence'](model,max_tokens=1024,session_id='S-EMBER-text-only')
    saved=model.model.rope_deltas;first=[];sync();began=time.monotonic()
    def deliver(item):
        if not first:first.append(time.monotonic()-began)
        callback(item)
    try:
        with env['fp32_reference']():
            for start in range(0,ids.shape[1],512):
                end=min(start+512,ids.shape[1])
                logits=seq.append(input_ids=ids[:,start:end].to(device),
                    logits_mode='last' if end==ids.shape[1] else 'none')
            sync();prefilled=time.monotonic()
            eos=model.generation_config.eos_token_id
            stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(processor.tokenizer.eos_token_id)
            result=emit_decode(seq,logits,processor,limit,stops,deliver);sync();ended=time.monotonic();size=cache_bytes(seq)
    finally:
        seq.reset();model.model.rope_deltas=saved
    assert seq.seen_tokens==0
    return dict(**result,QA_total_seconds=ended-began,prefill_seconds=prefilled-began,
        decode_seconds=ended-prefilled,first_token_seconds=first[0],input_tokens=ids.shape[1],
        QA_cache_bytes=size,branch_released=True,visual_tokens=0,vision_calls=0)
