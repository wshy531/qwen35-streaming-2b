"""Same joint input/positions, independently formed recent hybrid state.

Only query-time transient caches. No permanent state or memory bank mutation.
Not an equivalence-preserving change to joint model inference.
"""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
for name in ['2b-resolution-cost-v1','2b-protection-prototypes-v1']:
    sys.path.insert(0,str(R/'experiments'/name))
import torch
from resolution_runtime import context
from resolution_nodes import digest
from protected_cache import compose

@torch.inference_mode()
def answer(env,model,builder,history,recent,question,*,isolated=True,limit=8):
    if not recent:raise ValueError('nonempty current evidence required')
    if not 1<=limit<=16:raise ValueError('fixed output limit')
    nodes=history+recent;before=digest(nodes);saved=model.model.rope_deltas;owned=[]
    try:
        with env['fp32_reference']():
            emb,pos,ids,meta=context(model,builder,nodes,question,max_new_tokens=limit)
            prefix=builder.prefix_ids.shape[1];suffix=meta['suffix_tokens'];end=emb.shape[1]-suffix
            if history:
                _,_,hids,hmeta=context(model,builder,history,question,max_new_tokens=limit)
                boundary=hids.shape[1]-hmeta['suffix_tokens']
                assert torch.equal(ids[:,:boundary],hids[:,:boundary])
            else:boundary=prefix
            base=env['HybridSequence'](model,max_tokens=4096,session_id='shared-prefix');owned.append(base)
            base.append(inputs_embeds=emb[:,:prefix],position_ids=pos[:,:,:prefix],logits_mode='none')
            def append_segment(seq,lo,hi,local=False,last=False):
                result=None
                for start in range(lo,hi,512):
                    stop=min(start+512,hi);p=pos[:,:,start:stop].clone()
                    if local:p[0,0]=torch.arange(seq.seen_tokens,seq.seen_tokens+stop-start,device=p.device)
                    result=seq.append(inputs_embeds=emb[:,start:stop],position_ids=p,
                        logits_mode='last' if last and stop==hi else 'none')
                return result
            if isolated and history:
                hist=base.fork('history');rec=base.fork('recent');owned.extend([hist,rec])
                append_segment(hist,prefix,boundary)
                append_segment(rec,boundary,end,local=True)
                branch=compose(hist,rec,prefix);owned.append(branch)
                assert branch.seen_tokens==end
            else:
                branch=base.fork('joint');owned.append(branch);append_segment(branch,prefix,end)
            logits=append_segment(branch,end,emb.shape[1],last=True)
            first_logits=logits.detach().cpu().clone()
            eos=model.generation_config.eos_token_id
            stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
            generated,reason=env['greedy_decode'](branch,logits,limit,stops)
            cache_bytes=sum(x.numel()*x.element_size() for seq in owned for f in ('key_cache','value_cache','conv_states','recurrent_states') for x in getattr(seq.cache,f) if x is not None)
    finally:
        for seq in owned:seq.cache.reset()
        model.model.rope_deltas=saved
    assert digest(nodes)==before
    return dict(text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated,
        first_logits=first_logits,input_ids=ids,position_ids=pos.cpu(),reason=reason,temporary_cache_bytes_at_end=cache_bytes,
        read_state_unchanged=True,branch_released=True,isolated_recent=isolated,**meta)
