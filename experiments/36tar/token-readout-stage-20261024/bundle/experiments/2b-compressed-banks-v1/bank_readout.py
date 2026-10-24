"""Explicit-budget fresh readout composition, reusing the pinned old layout.

No silent question/history truncation, model loading, or default-path changes.
Compressed destination positions remain the existing heuristic, not a new
position-correctness claim. Actual runtime generation is deliberately separate.
"""
from dataclasses import dataclass
import itertools

import torch

from legacy_sources import load


@dataclass(frozen=True)
class Limits:
    context_tokens:int
    visual_tokens:int
    nodes:int
    suffix_tokens:int=128

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in (self.context_tokens,self.visual_tokens,self.nodes,self.suffix_tokens)):
            raise ValueError('explicit positive token/node bounds required')


@torch.inference_mode()
def context(model,builder,nodes,question,*,limits,max_new_tokens=16):
    if not nodes or len(nodes)>limits.nodes or sum(len(n.features) for n in nodes)>limits.visual_tokens:
        raise ValueError('declared visual/node budget exceeded; history was not truncated')
    if not 1<=max_new_tokens<=24:raise ValueError('explicit generation budget required')
    if any(a.last>=b.first or a.end>=b.begin for a,b in zip(nodes,nodes[1:])):
        raise ValueError('causal nonoverlapping node order required')
    validate=load().forest.validate
    offset=builder.prefix_ids.shape[1];all_ids=[builder.prefix_ids]
    positions=[torch.arange(offset).expand(3,-1)]
    for node in nodes:
        validate(node)
        if node.features.shape[1]!=model.get_input_embeddings().embedding_dim:
            raise ValueError('feature projection width differs from this model')
        label=f'<{node.center:.1f} seconds>' if node.first == node.last else f'<{node.begin:.1f}-{node.end:.1f} seconds, compressed memory>'
        ids=builder.tokenize(label+builder.processor.vision_start_token+builder.processor.video_token*len(node.features)+builder.processor.vision_end_token)
        coords=[]
        for modality,group in itertools.groupby((ids[0]==builder.processor.video_token_id).tolist()):
            size=len(list(group))
            if modality:
                if size!=len(node.features):raise ValueError('placeholder/feature mismatch')
                coords.extend((offset+int(source)-node.first,offset+int(y),offset+int(x)) for source,y,x in node.coords)
                offset+=max(8,node.last-node.first+1)
            else:
                coords.extend((offset+i,)*3 for i in range(size));offset+=size
        all_ids.append(ids);positions.append(torch.tensor(coords).T)
    suffix=builder.question_ids(question)
    if suffix.shape[1]>limits.suffix_tokens:
        raise ValueError('full question suffix exceeds declared limit; no truncation')
    all_ids.append(suffix);positions.append(torch.arange(offset,offset+suffix.shape[1]).expand(3,-1))
    ids=torch.cat(all_ids,1)
    if ids.shape[1]+max_new_tokens>limits.context_tokens:
        raise ValueError('full context exceeds declared limit; no truncation')
    four=torch.cat((torch.arange(ids.shape[1])[None,:],torch.cat(positions,1)),0)[:,None,:]
    device=model.get_input_embeddings().weight.device
    embeddings=model.get_input_embeddings()(ids.to(device))
    visual=torch.cat([n.features for n in nodes]).to(embeddings)
    mask=(ids.to(device)==builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
    if int(mask.sum())!=visual.numel():raise ValueError('visual insertion mismatch')
    return embeddings.masked_scatter(mask,visual),four.to(device),ids,dict(
        input_tokens=ids.shape[1],visual_tokens=len(visual),nodes=[n.metadata() for n in nodes],
        suffix_tokens=suffix.shape[1],source_time_rank_preserved=True,language_axis_contiguous=True,
        compressed_destination_coordinates_are_existing_heuristic=True,
        no_question_or_history_truncation=True,model_forward_performed=False)
