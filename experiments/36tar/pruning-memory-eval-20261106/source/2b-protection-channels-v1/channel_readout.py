"""Four combinations of full-attention KV and GDN state at the question boundary."""
import sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-isolated-recent-v1'))
sys.path.insert(0,str(R/'experiments/2b-protection-prototypes-v1'))
from experiment import context,digest
from protected_cache import compose
from channel_intervention import cross_channels
import torch
ARMS=('joint','isolated','joint_KV_recent_GDN','isolated_KV_joint_GDN')

@torch.inference_mode()
def answer_four(env,model,builder,h,r,question,limit=16):
    assert 1<=limit<=16 and r
    nodes=h+r;before=digest(nodes);saved=model.model.rope_deltas;owned=[];answers={}
    try:
        with env['fp32_reference']():
            emb,pos,ids,meta=context(model,builder,nodes,question,max_new_tokens=limit)
            prefix=builder.prefix_ids.shape[1];end=emb.shape[1]-meta['suffix_tokens']
            _,_,hid,hm=context(model,builder,h,question,max_new_tokens=limit) if h else (None,None,None,None)
            boundary=hid.shape[1]-hm['suffix_tokens'] if h else prefix
            if h:assert torch.equal(ids[:,:boundary],hid[:,:boundary])
            base=env['HybridSequence'](model,max_tokens=4096);owned.append(base)
            base.append(inputs_embeds=emb[:,:prefix],position_ids=pos[:,:,:prefix],logits_mode='none')
            def append(seq,lo,hi,local=False,last=False):
                result=None
                for begin in range(lo,hi,512):
                    stop=min(begin+512,hi);p=pos[:,:,begin:stop].clone()
                    if local:p[0,0]=torch.arange(seq.seen_tokens,seq.seen_tokens+stop-begin,device=p.device)
                    result=seq.append(inputs_embeds=emb[:,begin:stop],position_ids=p,logits_mode='last' if last and stop==hi else 'none')
                return result
            joint=base.fork('joint');owned.append(joint);append(joint,prefix,end)
            if h:
                hist=base.fork('history');rec=base.fork('recent');owned.extend([hist,rec])
                append(hist,prefix,boundary);append(rec,boundary,end,local=True)
                iso=compose(hist,rec,prefix);owned.append(iso)
            else:iso=joint
            assert iso.seen_tokens==joint.seen_tokens==end
            choices=((joint,joint),(iso,iso),(joint,iso),(iso,joint))
            for name,(attention,linear) in zip(ARMS,choices):
                branch=cross_channels(attention,linear);owned.append(branch)
                logits=append(branch,end,emb.shape[1],last=True)
                assert torch.isfinite(logits).all()
                eos=model.generation_config.eos_token_id
                stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
                tokens,reason=env['greedy_decode'](branch,logits,limit,stops)
                answers[name]=dict(text=builder.processor.tokenizer.decode(tokens,skip_special_tokens=True),generated_ids=tokens,
                    bounded_context_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                    position_sha256=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest(),reason=reason,**meta)
                branch.cache.reset()
    finally:
        for seq in owned:seq.cache.reset()
        model.model.rope_deltas=saved
    assert digest(nodes)==before
    return answers
