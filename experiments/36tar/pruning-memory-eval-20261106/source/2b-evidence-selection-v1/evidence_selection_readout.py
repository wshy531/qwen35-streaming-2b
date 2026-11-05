"""Whole-block readout with a unique module name and anchored coordinates."""
import torch
from selection import subset_columns

def compose(model,builder,all_nodes,selected_history_indices,*,allow_full=False,question):
    from sember_runtime import context
    from sember_nodes import digest
    history=all_nodes[:-4];recent=all_nodes[-4:]
    if len(recent)!=4 or any(len(n.features)!=108 or not n.metadata()['exact_pair'] for n in recent):
        raise ValueError('complete unmodified recent suffix required')
    if selected_history_indices!=sorted(set(selected_history_indices)):
        raise ValueError('history selection unique ordered')
    if any(type(i)!=int or not 0<=i<len(history) for i in selected_history_indices):
        raise ValueError('history selection outside bank')
    if len(selected_history_indices)>2 and not allow_full:
        raise ValueError('two-node first-version selection cap')
    if sum(len(history[i].features) for i in selected_history_indices)>432:
        raise ValueError('historical token cap')
    before=digest(all_nodes)
    emb,pos,ids,meta=context(model,builder,all_nodes,question,visual_limit=864,
                           context_limit=8192,max_new_tokens=8)
    prefix=builder.prefix_ids.shape[1];lengths=[]
    for n in all_nodes:
        label=f'<{n.center:.1f} seconds>' if n.first==n.last else f'<{n.begin:.1f}-{n.end:.1f} seconds, compressed memory>'
        lengths.append(builder.tokenize(label+builder.processor.vision_start_token+
            builder.processor.video_token*len(n.features)+builder.processor.vision_end_token).shape[1])
    selected=selected_history_indices+list(range(len(history),len(all_nodes)))
    cols=list(range(prefix))+[prefix+i for i in subset_columns(lengths,selected)]+list(range(prefix+sum(lengths),ids.shape[1]))
    index=torch.tensor(cols,dtype=torch.long,device=emb.device)
    out_emb=emb.index_select(1,index);out_pos=pos.index_select(2,index).clone();out_ids=ids[:,cols]
    out_pos[0,0]=torch.arange(len(cols),device=pos.device)
    assert digest(all_nodes)==before
    return out_emb,out_pos,out_ids,dict(meta,input_tokens=len(cols),
        visual_tokens=sum(len(all_nodes[i].features) for i in selected),
        nodes=[all_nodes[i].metadata() for i in selected],
        selected_history_indices=selected_history_indices,full_context_columns=cols,
        mrope_anchored_to_full=True,cache_indices_compacted=True,bank_unchanged=True)
