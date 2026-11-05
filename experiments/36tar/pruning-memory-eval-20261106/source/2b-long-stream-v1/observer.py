"""Incremental pair encoding, persistent hybrid state and isolated live readout.

No weight loading, scheduling or resource submission happens in this module.
The caller supplies the verified model and runtime. Each pair is encoded once;
the same computed projection updates the optional bank and hybrid main state.
"""
import time

import torch

from online_bank import OnlineBank


def cache_bytes(sequence):
    return sum(t.numel()*t.element_size() for name in
        ('key_cache','value_cache','conv_states','recurrent_states')
        for t in getattr(sequence.cache,name) if t is not None)


@torch.inference_mode()
def emit_decode(branch,logits,processor,max_new_tokens,eos_ids,on_token):
    if not callable(on_token):raise ValueError('incremental token callback required')
    if type(max_new_tokens) is not int or not 1<=max_new_tokens<=24:raise ValueError('bounded generation required')
    tokens=[];reason='budget'
    for i in range(max_new_tokens):
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite logits')
        token=int(logits[0,-1].argmax());tokens.append(token)
        on_token(dict(token_index=i,token_id=token,
            cumulative_text=processor.tokenizer.decode(tokens,skip_special_tokens=True),
            monotonic_s=time.monotonic(),is_stop=token in eos_ids))
        if token in eos_ids:reason='eos';break
        if i+1<max_new_tokens:
            logits=branch.append(input_ids=torch.tensor([[token]],device=logits.device),logits_mode='last')
    return dict(text=processor.tokenizer.decode(tokens,skip_special_tokens=True),
        generated_ids=tokens,stop_reason=reason)


class Observer:
    def __init__(self,env,model,builder,*,policy=None,main_tokens=2048,question_tokens=128,answer_tokens=8):
        if model.training or next(model.parameters()).dtype!=torch.float32:raise ValueError('FP32 eval required')
        if policy not in (None,'M0','M2'):raise ValueError('fixed policy required')
        if any(type(v) is not int or v<1 for v in (main_tokens,question_tokens,answer_tokens)):
            raise ValueError('positive budgets required')
        if main_tokens-question_tokens-answer_tokens<=builder.prefix_ids.shape[1]:raise ValueError('no input room')
        self.env,self.model,self.builder=env,model,builder
        self.main=env['HybridSequence'](model,max_tokens=main_tokens)
        self.bank=OnlineBank(policy) if policy else None
        self.main_budget=main_tokens-question_tokens-answer_tokens
        self.question_tokens,self.answer_tokens=question_tokens,answer_tokens
        self.pairs=self.resets=self.vision_encodings=0
        self.last_frame_id=self.last_pts=self.memory_from_pts=None
        self.invalid=False

    @torch.inference_mode()
    def append_pair(self,first,second):
        if self.invalid:raise RuntimeError('invalid observer')
        if (first.frame_id>=second.frame_id or first.timestamp>=second.timestamp or
            (self.last_frame_id is not None and (first.frame_id<=self.last_frame_id or first.timestamp<=self.last_pts))):
            raise ValueError('new strictly causal real frame pair required')
        prefix=self.builder.prefix_ids.shape[1] if self.main.seen_tokens==0 else 0
        inputs,record=self.builder.pair(first,second,self.main.seen_tokens+prefix,self.main.next_rope_position+prefix)
        n=inputs.input_ids.shape[1]
        if self.builder.prefix_ids.shape[1]+n>self.main_budget:raise ValueError('pair cannot fit fresh epoch')
        if self.main.seen_tokens+prefix+n>self.main_budget:
            self.main.reset();self.resets+=1;self.memory_from_pts=first.timestamp
            prefix=self.builder.prefix_ids.shape[1]
            inputs,record=self.builder.pair(first,second,prefix,prefix)
        if self.main.seen_tokens==0:self.memory_from_pts=first.timestamp
        device=self.model.get_input_embeddings().weight.device
        started=time.monotonic()
        try:
            with self.env['fp32_reference']():
                inputs=inputs.to(device)
                vision=self.model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True)
                features=torch.cat(vision.pooler_output,0)
                embeddings=self.model.get_input_embeddings()(inputs.input_ids)
                mask=(inputs.input_ids==self.builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
                embeddings=embeddings.masked_scatter(mask,features.to(embeddings))
                if prefix:self.main.append(input_ids=self.builder.prefix_ids.to(device),logits_mode='none')
                self.main.append(inputs_embeds=embeddings,position_ids=inputs.position_ids,logits_mode='none')
                state=None if self.bank is None else self.bank.append(features,self.pairs,first.timestamp,second.timestamp)
                self.vision_encodings+=1;self.pairs+=1
        except BaseException:
            self.invalid=True;raise
        self.last_frame_id,self.last_pts=second.frame_id,second.timestamp
        return dict(record=record,seconds=time.monotonic()-started,bank=state,**self.status())

    def status(self):
        return dict(pairs=self.pairs,resets=self.resets,vision_encodings=self.vision_encodings,
            last_frame_id=self.last_frame_id,last_pts=self.last_pts,memory_from_pts=self.memory_from_pts,
            main_seen_tokens=self.main.seen_tokens,main_next_rope=self.main.next_rope_position,
            main_cache_bytes=cache_bytes(self.main),invalid=self.invalid)

    @torch.inference_mode()
    def ask(self,question,*,read_from,on_token):
        if self.invalid or not self.pairs:raise RuntimeError('valid observed pair required')
        if not callable(on_token):raise ValueError('incremental token callback required')
        if read_from not in ('main','bank','text'):raise ValueError('explicit read source required')
        if read_from=='bank' and self.bank is None:raise ValueError('no external bank')
        suffix=self.builder.question_ids(question)
        if suffix.shape[1]>self.question_tokens:raise ValueError('full question exceeds reserve')
        before=self.status();bank_hash=None if self.bank is None else self.bank.fingerprint()
        saved=self.model.model.rope_deltas;branch=None;started=time.monotonic();first_time=None
        visible_time=None;last_time=None
        def deliver(item):
            nonlocal first_time,visible_time,last_time
            if first_time is None:first_time=item['monotonic_s']
            if visible_time is None and item['cumulative_text']:visible_time=item['monotonic_s']
            last_time=item['monotonic_s'];on_token(item)
        try:
            with self.env['fp32_reference']():
                if read_from=='main':
                    branch=self.main.fork()
                    logits=branch.append(input_ids=suffix.to(self.model.get_input_embeddings().weight.device),logits_mode='last')
                    meta=dict(input_tokens=branch.seen_tokens,visual_tokens=None)
                else:
                    from readout import context,Limits
                    if read_from=='bank':
                        emb,pos,_,meta=context(self.model,self.builder,self.bank.export(),question,
                            limits=Limits(8192,384,8,self.question_tokens),max_new_tokens=self.answer_tokens)
                    else:
                        from capability_pilot import text_context
                        emb,pos,_,meta=text_context(self.env,self.model,self.builder,question)
                    branch=self.env['HybridSequence'](self.model,max_tokens=8192,session_id='historical-qa')
                    for i in range(0,emb.shape[1],512):
                        stop=min(i+512,emb.shape[1])
                        logits=branch.append(inputs_embeds=emb[:,i:stop],position_ids=pos[:,:,i:stop],
                            logits_mode='last' if stop==emb.shape[1] else 'none')
                configured=self.model.generation_config.eos_token_id
                stops=set(configured if isinstance(configured,list) else [configured])
                stops.add(self.builder.processor.tokenizer.eos_token_id)
                result=emit_decode(branch,logits,self.builder.processor,self.answer_tokens,stops,deliver)
        finally:
            if branch is not None:branch.cache.reset()
            self.model.model.rope_deltas=saved
        assert self.status()==before,'question changed observation counters'
        assert self.bank is None or self.bank.fingerprint()==bank_hash,'question changed bank'
        return dict(**result,read_from=read_from,input_tokens=meta['input_tokens'],
            visual_tokens=meta['visual_tokens'],first_token_seconds=first_time-started,
            first_visible_text_seconds=None if visible_time is None else visible_time-started,
            last_token_seconds=last_time-started,total_seconds=time.monotonic()-started,
            observation_through_s=self.last_pts,bank_unchanged=True,
            main_counters_unchanged=True)

    def close(self):
        self.main.reset();self.bank=None;self.invalid=True
