"""Frozen recent-read representation; no answer, classifier prompt or old margins."""

def inputs(model, builder, nodes, question, evidence):
    import torch
    from sember_runtime import context
    if len(nodes)!=4 or any(n.first!=n.last or len(n.features)!=108 for n in nodes):
        raise ValueError('four exact complete pairs required')
    emb,pos,ids,meta=context(model,builder,nodes,question,visual_limit=432,
        context_limit=2048,suffix_limit=1024,max_new_tokens=8)
    if not evidence:
        start=builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_start_token)
        end=builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_end_token)
        keep=~((ids[0]==builder.processor.video_token_id)|(ids[0]==start)|(ids[0]==end))
        ids=ids[:,keep];emb=emb[:,keep.to(emb.device)]
        pos=torch.arange(ids.shape[1],device=emb.device)[None,None,:].expand(4,1,-1).clone()
    return emb,pos,ids,dict(input_tokens=ids.shape[1],
        visual_tokens=int((ids==builder.processor.video_token_id).sum()),
        uses_options=False,uses_history=False,uses_classification_prompt=False)

def extract(env, model, builder, nodes, question, evidence):
    torch=env['torch']
    from sember_nodes import digest
    before=digest(nodes);saved=model.model.rope_deltas;handle=None;sequence=None
    captured=[]
    def last_position(module, args, output):
        captured[:] = [output[:,-1,:].detach().float().cpu().clone()]
    try:
        with torch.inference_mode(),env['fp32_reference']():
            emb,pos,ids,meta=inputs(model,builder,nodes,question,evidence)
            handle=model.model.language_model.norm.register_forward_hook(last_position)
            sequence=env['HybridSequence'](model,max_tokens=2048,session_id='read-benefit-feature')
            for begin in range(0,emb.shape[1],512):
                end=min(begin+512,emb.shape[1])
                result=sequence.append(inputs_embeds=emb[:,begin:end],position_ids=pos[:,:,begin:end],logits_mode='none')
                if result is not None:
                    raise RuntimeError('unneeded vocabulary logits computed')
            if len(captured)!=1 or captured[0].shape!=(1,model.get_input_embeddings().embedding_dim) or not torch.isfinite(captured[0]).all():
                raise ValueError('finite final normalized last-position hidden state required')
            return captured[0][0],meta
    finally:
        if handle is not None:handle.remove()
        if sequence is not None:sequence.cache.reset()
        model.model.rope_deltas=saved
        if digest(nodes)!=before:raise RuntimeError('read representation changed visual evidence')
