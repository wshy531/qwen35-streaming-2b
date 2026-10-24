"""Isolated original-model reference; only the arrived complete-pair RGB prefix.

The reference deliberately rereads its CPU RGB history at every timed question.
Those retained pixels and all preprocessing/prefill costs are reported. It does
not share the bounded candidate's process or bank. No model loading here.
"""
import time

import torch

from observer import emit_decode


def layout_for(model,builder,packets,metadata,question,grid,device,guard=32768):
    processor=builder.processor
    timestamps=processor._calculate_timestamps([p.frame_id for p in packets],metadata['fps'],2)
    per_pair=int(grid[0,1:].prod())//processor.video_processor.merge_size**2
    visual=''.join(f'<{t:.1f} seconds>'+processor.vision_start_token+processor.video_token*per_pair+
        processor.vision_end_token for t in timestamps)
    ids=builder.tokenize(builder.template(question).replace(builder.marker,visual)).to(device)
    mm=torch.zeros_like(ids);mm[ids==processor.video_token_id]=2
    oracle=torch.repeat_interleave(grid,grid[:,0],0).clone();oracle[:,0]=1
    coords,delta=model.model.get_rope_index(input_ids=ids,mm_token_type_ids=mm,video_grid_thw=oracle)
    n=ids.shape[1];pos=torch.cat((torch.arange(n,device=device)[None,None,:],coords),0)
    assert n+8<=guard<=model.config.text_config.max_position_embeddings
    assert int(pos.max())<model.config.text_config.max_position_embeddings
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),mm_token_type_ids=mm,
        video_grid_thw=grid,position_ids=pos),delta


def chunked_inputs_for(env,model,builder,packets,metadata,question,device,guard=32768):
    """Same native full model input, with pair-sized CPU pixel preprocessing.

    No vision features/KV are computed here. Pair pixels are concatenated in
    chronological order for ONE original full-video forward. Short full-call
    bitwise pixel/layout equality is required in CPU preflight.
    """
    if not packets or len(packets)%2:raise ValueError('complete real pairs required')
    pixels=[];grid=None
    for i in range(0,len(packets),2):
        pair=packets[i:i+2]
        item=builder.processor.video_processor(videos=[env['np'].stack([p.rgb for p in pair])],
            video_metadata=[dict(metadata,frames_indices=[p.frame_id for p in pair])],
            do_resize=False,do_sample_frames=False,return_tensors='pt')
        g=item['video_grid_thw']
        assert g.shape==(1,3) and int(g[0,0])==1
        if grid is None:grid=g.clone()
        else:assert torch.equal(g[:,1:],grid[:,1:])
        pixels.append(item['pixel_values_videos'])
    grid[0,0]=len(packets)//2;grid=grid.to(device)
    inputs,delta=layout_for(model,builder,packets,metadata,question,grid,device,guard)
    inputs['pixel_values_videos']=torch.cat(pixels,0).to(device)
    meta=dict(input_tokens=inputs['input_ids'].shape[1],visual_tokens=int((inputs['input_ids']==builder.processor.video_token_id).sum()),
        full_context_guard=guard,position_adapter='inherited official per-temporal-group oracle grid',
        retained_RGB_bytes=sum(p.rgb.nbytes for p in packets),real_frames=len(packets),
        last_frame_id=packets[-1].frame_id,observation_through_s=packets[-1].timestamp,
        pixel_preprocess_chunk_pairs=1,native_whole_model_prefill=True)
    return inputs,delta,meta


def inputs_for(env,model,builder,packets,metadata,question,device,guard=32768):
    if not packets or len(packets)%2:raise ValueError('nonempty complete real pairs required')
    inputs=builder.processor(text=[builder.template(question)],
        videos=[env['np'].stack([p.rgb for p in packets])],
        video_metadata=[dict(metadata,frames_indices=[p.frame_id for p in packets])],
        do_resize=False,do_sample_frames=False,return_tensors='pt').to(device)
    n=inputs.input_ids.shape[1]
    assert n+8<=guard<=model.config.text_config.max_position_embeddings,'explicit guard exceeded; no truncation'
    oracle=torch.repeat_interleave(inputs.video_grid_thw,inputs.video_grid_thw[:,0],0).clone()
    oracle[:,0]=1
    coords,delta=model.model.get_rope_index(**dict(inputs,video_grid_thw=oracle))
    inputs['position_ids']=torch.cat((torch.arange(n,device=device)[None,None,:],coords),0)
    assert int(inputs.position_ids.max())<model.config.text_config.max_position_embeddings
    return inputs,delta,dict(input_tokens=n,
        visual_tokens=int((inputs.input_ids==builder.processor.video_token_id).sum()),
        full_context_guard=guard,position_adapter='inherited official per-temporal-group oracle grid',
        retained_RGB_bytes=sum(p.rgb.nbytes for p in packets),real_frames=len(packets),
        last_frame_id=packets[-1].frame_id,observation_through_s=packets[-1].timestamp)


class SingleTokenCursor:
    def __init__(self,model,result,inputs):
        self.model=model;self.cache=result.past_key_values;self.length=inputs['input_ids'].shape[1]
        self.rope=int(inputs['position_ids'][1:].max())+1
    def append(self,*,input_ids,logits_mode):
        assert input_ids.shape==(1,1) and logits_mode=='last'
        device=input_ids.device
        pos=torch.tensor([[[self.length]],[[self.rope]],[[self.rope]],[[self.rope]]],device=device)
        result=self.model(input_ids=input_ids,position_ids=pos,
            cache_position=torch.tensor([self.length],device=device),
            attention_mask=torch.ones((1,self.length+1),dtype=torch.long,device=device),
            past_key_values=self.cache,use_cache=True,logits_to_keep=1)
        self.cache=result.past_key_values;self.length+=1;self.rope+=1
        return result.logits


class NativeObserver:
    def __init__(self,env,model,builder,source,*,preflight=False):
        self.env,self.model,self.builder,self.source=env,model,builder,source
        self.packets=[];self.last_pts=None;self.pairs=0;self.preflight=preflight
        self.contexts=[]
    def append_pair(self,a,b):
        assert self.last_pts is None or self.last_pts<a.timestamp<b.timestamp
        self.packets.extend((a,b));self.pairs+=1;self.last_pts=b.timestamp
        return self.status()
    def status(self):return dict(pairs=self.pairs,last_pts=self.last_pts,
        last_frame_id=None if not self.packets else self.packets[-1].frame_id,
        retained_RGB_bytes=sum(p.rgb.nbytes for p in self.packets))

    @torch.inference_mode()
    def ask(self,question,*,read_from,on_token):
        assert read_from=='native'
        started=time.monotonic();device=next(self.model.parameters()).device
        saved=self.model.model.rope_deltas;cursor=None
        inputs,delta,meta=chunked_inputs_for(self.env,self.model,self.builder,self.packets,
            self.source.metadata,question,device)
        if self.preflight:
            self.contexts.append(meta)
            return dict(status='processor_and_positions_only_NOT_model_forward',**meta)
        first=visible=last=None
        def deliver(item):
            nonlocal first,visible,last
            now=item['monotonic_s']
            if first is None:first=now
            if visible is None and item['cumulative_text']:visible=now
            last=now;on_token(item)
        try:
            with self.env['fp32_reference']():
                self.model.model.rope_deltas=delta
                result=self.model(**inputs,use_cache=True,logits_to_keep=1)
                cursor=SingleTokenCursor(self.model,result,inputs)
                configured=self.model.generation_config.eos_token_id
                stops=set(configured if isinstance(configured,list) else [configured])
                stops.add(self.builder.processor.tokenizer.eos_token_id)
                answer=emit_decode(cursor,result.logits,self.builder.processor,8,stops,deliver)
        finally:
            # Original Qwen cache has no reset() API. Drop owned tensor lists;
            # this branch is disposable and is never the observation state.
            if cursor is not None:
                for name in ('key_cache','value_cache','conv_states','recurrent_states'):
                    values=getattr(cursor.cache,name)
                    for i in range(len(values)):values[i]=None
            self.model.model.rope_deltas=saved
        return dict(**answer,**meta,read_from='native',first_token_seconds=first-started,
            first_visible_text_seconds=None if visible is None else visible-started,
            last_token_seconds=last-started,total_seconds=time.monotonic()-started)
    def close(self):self.packets.clear()
