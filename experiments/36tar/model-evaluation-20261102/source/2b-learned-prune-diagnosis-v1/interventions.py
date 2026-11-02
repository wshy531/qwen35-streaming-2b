"""Independent selection/bias interventions; original experiment is immutable."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-learned-prune-main-v1'))
from protocol import GroupScorer, runtime, make_builder, context, question_text
from bias_prune import biased_block

ARMS={'initial_original':('initial','initial'),
      'final_original':('final','final'),
      'initial_none':('initial','none'),
      'final_none':('final','none'),
      'initial_final_bias':('initial','final'),
      'final_initial_bias':('final','initial')}

def encode_all(visual,pixels,grid,scorers,previous=None):
    assert grid.tolist()==[[1,18,24]] and len(visual.blocks)==24
    x=visual.patch_embed(pixels)+visual.fast_pos_embed_interpolate(grid)
    rope=visual.rot_pos_emb(grid).reshape(len(x),-1)
    emb=torch.cat((rope,rope),-1);pos=(emb.cos(),emb.sin())
    cu=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
    for block in visual.blocks[:11]:
        x=block(x,cu_seqlens=cu,position_embeddings=pos)
    logits={};selected={};state=None
    for name,scorer in scorers.items():
        logits[name],state=scorer(x,previous)
        selected[name]=torch.argsort(logits[name],descending=True,stable=True)[:54].sort().values
    # Bias and selected indices are computed from the identical pre-intervention x.
    variants={name:biased_block(visual.blocks[11],x,pos,s) for name,s in logits.items()}
    variants['none']=visual.blocks[11](x,cu_seqlens=cu,position_embeddings=pos)
    output={}
    for arm,(selection,bias) in ARMS.items():
        ids=selected[selection];patches=(ids[:,None]*4+torch.arange(4,device=x.device)).flatten()
        y=variants[bias][patches];p=tuple(z[patches] for z in pos)
        short=torch.tensor([0,len(y)],device=x.device,dtype=torch.int32)
        for block in visual.blocks[12:]:y=block(y,cu_seqlens=short,position_embeddings=p)
        output[arm]=(visual.merger(y),ids)
    stats={'selection_overlap':len(set(selected['initial'].tolist())&set(selected['final'].tolist()))}
    for name,s in logits.items():
        b=torch.nn.functional.logsigmoid(s.detach())
        stats[name]={'bias_min':float(b.min()),'bias_max':float(b.max()),
                     'bias_span':float(b.max()-b.min()),'below_minus2':int((b < -2).sum()),
                     'selected':selected[name].tolist()}
    return output,state,stats

def language_compare(env,model,builder,records,q,chunk=512):
    """Use identical embeddings, positions and weights; measure the entire vocabulary."""
    emb,pos,ids=context(model,builder,records,question_text(q))
    raw=model.model.language_model(inputs_embeds=emb,position_ids=pos,
          attention_mask=torch.ones(emb.shape[:2],device=emb.device,dtype=torch.long),use_cache=False)
    full=model.lm_head(raw.last_hidden_state[:,-1:]).float()
    branch=env['HybridSequence'](model,max_tokens=4096,session_id='diagnosis')
    saved=model.model.rope_deltas
    try:
        for begin in range(0,emb.shape[1],chunk):
            end=min(begin+chunk,emb.shape[1])
            last=branch.append(inputs_embeds=emb[:,begin:end],position_ids=pos[:,:,begin:end],
                    logits_mode='last' if end==emb.shape[1] else 'none')
        last=last.reshape_as(full).float()
        delta=(full-last).abs()
        options=builder.tokenize('A').new_tensor([builder.tokenize(c).item() for c in 'ABCDE'])
        gold=builder.tokenize(chr(65+q['answer_index'])).to(full.device).reshape(-1)
        return {'tokens':ids.shape[1],'max_abs':float(delta.max()),
                'relative_l2':float(torch.linalg.vector_norm(full-last)/torch.linalg.vector_norm(full).clamp_min(1e-12)),
                'argmax_equal':bool(torch.equal(full.argmax(-1),last.argmax(-1))),
                'option_order_equal':bool(torch.equal(full[...,options].argsort(),last[...,options].argsort())),
                'full_CE':float(torch.nn.functional.cross_entropy(full.reshape(1,-1),gold)),
                'chunk_CE':float(torch.nn.functional.cross_entropy(last.reshape(1,-1),gold))}
    finally:
        branch.cache.reset();model.model.rope_deltas=saved
