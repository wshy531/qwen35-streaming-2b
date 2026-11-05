"""Shared diagnostic contexts, with raw history explicitly outside candidate budget."""
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
sys.path.insert(0,str(ROOT/'experiments/2b-long-stream-v1'))

import torch
from readout import context,Limits
from observer import emit_decode

ARMS=('A_raw_old','B_compressed_old','C_raw_old_recent','D_compressed_old_recent',
      'E_full_pairwise','F_bank_reproduction')
LIMITS=Limits(32768,16384,256,128)


def choices(raw,compressed,row):
    assert len(raw)==row['total_prefix_pairs']
    assert len(compressed)==len(row['expected_bank_nodes'])
    old=[raw[i] for i in row['raw_selected_pair_indices']]
    packed=[compressed[i] for i in row['selected_node_indices']]
    recent=[raw[i] for i in row['recent_pair_indices']]
    return dict(A_raw_old=old,B_compressed_old=packed,C_raw_old_recent=old+recent,
        D_compressed_old_recent=packed+recent,E_full_pairwise=raw,F_bank_reproduction=compressed)


@torch.inference_mode()
def answer(env,model,builder,nodes,question,on_token):
    saved=model.model.rope_deltas;branch=None;start=time.monotonic()
    try:
        with env['fp32_reference']():
            emb,pos,ids,meta=context(model,builder,nodes,question,limits=LIMITS,max_new_tokens=8)
            branch=env['HybridSequence'](model,max_tokens=32768,session_id='readability-diagnostic')
            for i in range(0,emb.shape[1],512):
                end=min(i+512,emb.shape[1])
                logits=branch.append(inputs_embeds=emb[:,i:end],position_ids=pos[:,:,i:end],
                    logits_mode='last' if end==emb.shape[1] else 'none')
            cfg=model.generation_config.eos_token_id
            eos=set(cfg if isinstance(cfg,list) else [cfg]);eos.add(builder.processor.tokenizer.eos_token_id)
            result=emit_decode(branch,logits,builder.processor,8,eos,on_token)
    finally:
        if branch is not None:branch.cache.reset()
        model.model.rope_deltas=saved
    return dict(**result,input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens'],
        nodes=meta['nodes'],seconds=time.monotonic()-start)
