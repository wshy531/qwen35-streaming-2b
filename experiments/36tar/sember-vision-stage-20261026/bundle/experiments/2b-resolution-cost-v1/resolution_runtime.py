# Isolated derivative; source and exact changes recorded in adapter_derivation.json.
"""Explicit high-resolution builder and fresh bank readout, preserving old kernels."""
import itertools
import math
import torch
from transformers.feature_extraction_utils import BatchFeature
from resolution_nodes import validate


def make_builder(env,processor,canvas=(288,384)):
    class Builder(env['PairInputBuilder']):
        def __init__(self):
            factor=processor.video_processor.patch_size*processor.video_processor.merge_size
            if canvas not in ((192,256),(288,384),(384,512)) or any(n%factor for n in canvas) or processor.video_processor.temporal_patch_size!=2:
                raise ValueError('registered pair canvas and official processor required')
            self.processor,self.canvas,self.resize_mode=processor,canvas,'letterbox'
            self.marker=processor.vision_start_token+processor.video_token+processor.vision_end_token
            self.prefix_text,_=self.template('probe').split(self.marker)
            self.prefix_ids=self.tokenize(self.prefix_text)
    return Builder()


@torch.inference_mode()
def context(model,builder,nodes,question,*,visual_limit=864,context_limit=4096,suffix_limit=512,max_new_tokens=16):
    if not nodes or len(nodes)>32 or sum(len(n.features) for n in nodes)>visual_limit:
        raise ValueError('explicit bounded visual/node budget exceeded')
    if any(a.last>=b.first or a.end>=b.begin for a,b in zip(nodes,nodes[1:])):
        raise ValueError('ordered nonoverlapping nodes required')
    offset=builder.prefix_ids.shape[1];ids_list=[builder.prefix_ids];positions=[torch.arange(offset).expand(3,-1)]
    for node in nodes:
        validate(node)
        if node.features.shape[1]!=model.get_input_embeddings().embedding_dim:raise ValueError('model projection width mismatch')
        label=f'<{node.center:.1f} seconds>' if node.first==node.last else f'<{node.begin:.1f}-{node.end:.1f} seconds, compressed memory>'
        ids=builder.tokenize(label+builder.processor.vision_start_token+builder.processor.video_token*len(node.features)+builder.processor.vision_end_token)
        coords=[]
        for modality,group in itertools.groupby((ids[0]==builder.processor.video_token_id).tolist()):
            count=len(list(group))
            if modality:
                if count!=len(node.features):raise ValueError('placeholder count mismatch')
                coords.extend((offset+int(source)-node.first,offset+int(y),offset+int(x)) for source,y,x in node.coords)
                offset+=max(*node.grid,node.last-node.first+1)
            else:coords.extend((offset+i,)*3 for i in range(count));offset+=count
        ids_list.append(ids);positions.append(torch.tensor(coords).T)
    suffix=builder.question_ids(question)
    if suffix.shape[1]>suffix_limit:raise ValueError('complete question exceeds reserved suffix; no truncation')
    ids_list.append(suffix);positions.append(torch.arange(offset,offset+suffix.shape[1]).expand(3,-1))
    ids=torch.cat(ids_list,1)
    if not 1<=max_new_tokens<=16 or ids.shape[1]+max_new_tokens>context_limit:raise ValueError('declared QA context exceeded')
    pos=torch.cat((torch.arange(ids.shape[1])[None,:],torch.cat(positions,1)),0)[:,None,:]
    device=model.get_input_embeddings().weight.device;emb=model.get_input_embeddings()(ids.to(device))
    visual=torch.cat([n.features for n in nodes]).to(emb);mask=(ids.to(device)==builder.processor.video_token_id).unsqueeze(-1).expand_as(emb)
    if int(mask.sum())!=visual.numel():raise ValueError('visual embedding insertion mismatch')
    return emb.masked_scatter(mask,visual),pos.to(device),ids,dict(input_tokens=ids.shape[1],visual_tokens=len(visual),
        suffix_tokens=suffix.shape[1],nodes=[n.metadata() for n in nodes],compressed_positions_are_heuristic=True)
