"""Actual recent visual prefill; no stored answers, history values or generation."""
from dataclasses import dataclass
import hashlib
import time
from gate_common import shared

def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

@dataclass(frozen=True)
class Packet:
    frame_id: int
    timestamp: float
    rgb: object

def recent_pairs(path, source):
    import av
    from PIL import Image
    import numpy as np
    # Same fixed canvas as the previous recorded experiment, without importing its orchestration.
    targets = {f['frame_id']: (p['index'], i, f) for p in source['pairs'] for i,f in enumerate(p['frames'])}
    if len(targets) != len(source['pairs'])*2: raise ValueError('duplicate recent frame')
    origin = None; pending = None; found = 0
    with av.open(str(path)) as c:
        stream = c.streams.video[0]; stream.codec_context.thread_count = 1
        for index, frame in enumerate(c.decode(stream)):
            if origin is None: origin = float(frame.time)
            if index not in targets: continue
            pair_index, member, target = targets[index]
            pts = float(frame.time)-origin
            image = frame.to_image()
            image.thumbnail((384,384),Image.Resampling.BICUBIC)
            scale = min(384/image.width,288/image.height)
            size = (max(1,min(384,round(image.width*scale))),max(1,min(288,round(image.height*scale))))
            image = image.resize(size,Image.Resampling.BICUBIC)
            canvas = Image.new('RGB',(384,288));canvas.paste(image,((384-size[0])//2,(288-size[1])//2))
            if abs(pts-target['pts_s'])>1e-7 or hashlib.sha256(canvas.tobytes()).hexdigest()!=target['mid_rgb_sha256']:
                raise ValueError('recent pixels or timestamp changed')
            packet = Packet(index,pts,np.asarray(canvas).copy());found += 1
            if member == 0:
                if pending is not None: raise ValueError('unfinished pair')
                pending = (pair_index,packet)
            else:
                if pending is None or pending[0]!=pair_index: raise ValueError('invalid pair order')
                yield pair_index,pending[1],packet
                pending = None
            if found == len(targets): break
    if found != len(targets) or pending is not None: raise ValueError('missing recent input')

def projected_pair(env, model, builder, first, second):
    torch=env['torch']; saved=model.model.rope_deltas
    with torch.inference_mode(),env['fp32_reference']():
        try:
            batch,_=builder.pair(first,second,0,0)
            batch=batch.to(model.get_input_embeddings().weight.device)
            vision=model.model.get_video_features(batch.pixel_values_videos,batch.video_grid_thw,return_dict=True)
            value=torch.cat(vision.pooler_output,0).detach().to(device='cpu',dtype=torch.float32)
            if value.shape != (108,model.get_input_embeddings().embedding_dim) or not torch.isfinite(value).all():
                raise ValueError('complete finite projected pair required')
            return value
        finally: model.model.rope_deltas=saved

def inputs(model, builder, nodes, prompt, evidence):
    import torch
    from sember_runtime import context
    from sember_nodes import digest
    if len(nodes)!=4 or any(len(n.features)!=108 or n.first!=n.last for n in nodes):
        raise ValueError('four exact real pairs required')
    before=digest(nodes)
    emb,pos,ids,meta=context(model,builder,nodes,prompt['text'],visual_limit=432,
        context_limit=2048,suffix_limit=1024,max_new_tokens=8)
    if not evidence:
        # Keep all textual context including frame times; omit all visual tokens/values.
        start_id=builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_start_token)
        end_id=builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_end_token)
        mask=~((ids[0]==builder.processor.video_token_id)|(ids[0]==start_id)|(ids[0]==end_id))
        ids=ids[:,mask];emb=emb[:,mask.to(emb.device)]
        pos=torch.arange(ids.shape[1],device=emb.device)[None,None,:].expand(4,1,-1).clone()
    codes=[builder.processor.tokenizer.encode(c,add_special_tokens=False) for c in prompt['codes']]
    if any(len(c)!=1 for c in codes) or codes[0]==codes[1]: raise ValueError('two single-token codes required')
    if digest(nodes)!=before: raise RuntimeError('selector mutated recent features')
    return emb,pos,ids,[c[0] for c in codes],dict(input_tokens=ids.shape[1],
        visual_tokens=int((ids==builder.processor.video_token_id).sum()),
        feature_digest=before,features_unchanged=True,**{'no_historical_features':True})

def score(env,model,builder,nodes,prompt,evidence):
    torch=env['torch'];device=model.get_input_embeddings().weight.device
    seq=None;saved=model.model.rope_deltas
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    sync();started=time.monotonic()
    try:
        with torch.inference_mode(),env['fp32_reference']():
            emb,pos,ids,codes,meta=inputs(model,builder,nodes,prompt,evidence)
            seq=env['HybridSequence'](model,max_tokens=2048,session_id='recent-evidence-query-local')
            for b in range(0,emb.shape[1],512):
                e=min(b+512,emb.shape[1])
                logits=seq.append(inputs_embeds=emb[:,b:e],position_ids=pos[:,:,b:e],
                    logits_mode='last' if e==emb.shape[1] else 'none')
            values=logits[0,-1,codes].float().detach().cpu()
            if not torch.isfinite(values).all():raise FloatingPointError('nonfinite adequacy logits')
            sync()
            return dict(view=prompt['view'],semantic_order=prompt['semantic_order'],candidate_logits=values.tolist(),
                input_ids_sha256=tensor_sha(ids),position_ids_sha256=tensor_sha(pos),
                seconds=time.monotonic()-started,branch_released=True,generated_tokens=0,**meta)
    finally:
        if seq is not None:seq.cache.reset()
        model.model.rope_deltas=saved
