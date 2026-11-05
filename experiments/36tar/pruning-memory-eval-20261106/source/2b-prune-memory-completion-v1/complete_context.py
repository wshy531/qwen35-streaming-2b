"""Uncompressed causal history reference, derived from resolution_runtime.context.

Same timestamp strings, raw spatial coordinates, prefix and suffix. Only the
explicit node/context/output capacity limits are extended. CPU parity with the
original bounded builder is required. This archive is NOT a bounded method.
"""
import itertools
import torch
from resolution_nodes import validate


@torch.inference_mode()
def full_context(model, builder, nodes, question, limit=128):
    if not nodes or len(nodes)>1024 or any(n.first!=n.last or len(n.features)!=108 for n in nodes):
        raise ValueError('complete original 108-token pairs only')
    if any(a.last>=b.first or a.end>=b.begin for a,b in zip(nodes,nodes[1:])):
        raise ValueError('ordered disjoint past pairs required')
    offset=builder.prefix_ids.shape[1]
    ids_list=[builder.prefix_ids]
    positions=[torch.arange(offset).expand(3,-1)]
    for node in nodes:
        validate(node)
        assert node.features.shape[1]==model.get_input_embeddings().embedding_dim
        ids=builder.tokenize(f'<{node.center:.1f} seconds>'+builder.processor.vision_start_token+
                             builder.processor.video_token*len(node.features)+builder.processor.vision_end_token)
        coords=[]
        for modality,group in itertools.groupby((ids[0]==builder.processor.video_token_id).tolist()):
            count=len(list(group))
            if modality:
                assert count==len(node.features)
                coords.extend((offset+int(source)-node.first,offset+int(y),offset+int(x)) for source,y,x in node.coords)
                offset+=max(*node.grid,node.last-node.first+1)
            else:
                coords.extend((offset+i,)*3 for i in range(count))
                offset+=count
        ids_list.append(ids)
        positions.append(torch.tensor(coords).T)
    suffix=builder.question_ids(question)
    if suffix.shape[1]>512:raise ValueError('no suffix truncation')
    ids_list.append(suffix)
    positions.append(torch.arange(offset,offset+suffix.shape[1]).expand(3,-1))
    ids=torch.cat(ids_list,1)
    if not 1<=limit<=128 or ids.shape[1]+limit>65536:raise ValueError('full-history capacity exceeded; no truncation')
    pos=torch.cat((torch.arange(ids.shape[1])[None,:],torch.cat(positions,1)),0)[:,None,:]
    device=model.get_input_embeddings().weight.device
    emb=model.get_input_embeddings()(ids.to(device))
    visual=torch.cat([n.features for n in nodes]).to(emb)
    mask=(ids.to(device)==builder.processor.video_token_id).unsqueeze(-1).expand_as(emb)
    assert int(mask.sum())==visual.numel()
    return emb.masked_scatter(mask,visual),pos.to(device),ids,dict(
        input_tokens=ids.shape[1],visual_tokens=len(visual),suffix_tokens=suffix.shape[1],
        nodes=[n.metadata() for n in nodes],complete_sampled_history=True,
        history_truncated=False,compressed_positions_are_heuristic=False)
