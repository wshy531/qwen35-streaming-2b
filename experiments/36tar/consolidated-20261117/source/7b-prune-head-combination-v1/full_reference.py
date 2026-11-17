"""Separate complete-history reference helper; never used by bounded banks.

Calls the existing official-language cached prefill in blocks of 512. limit=0
requests first logits only, without generating an answer or a hidden rationale.
Real-model qualification and resource sizing are required before full rollout.
"""
from combination import *

@torch.no_grad()
def score_complete(model,processor,nodes,q,shift,guard=lambda:None):
    before=digest(nodes)
    emb,pos,ids=segmented.context(model,processor,nodes,prompt(q,shift),'segmented')
    generated,logits=readout.generate_context(model,emb,pos,limit=0,guard=guard)
    assert not generated and digest(nodes)==before
    return logits[option_ids(processor,len(q['options']))],dict(input_tokens=ids.numel(),visual_tokens=sum(len(n.features) for n in nodes))
