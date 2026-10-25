# Isolated derivative; source and exact changes recorded in adapter_derivation.json.
"""Mutually exclusive bounded banks or full hybrid reference; isolated streamed QA."""
import hashlib
import time
import torch
from resolution_banks import Banks
from resolution_nodes import digest
from resolution_runtime import context
from resolution_common import ARMS


def cache_tensors(sequence):
    return [t for field in ('key_cache','value_cache','conv_states','recurrent_states')
        for t in getattr(sequence.cache,field) if t is not None]


def signature(sequence):
    # Inference tensors have no usable version counter; compare their actual bytes.
    h=hashlib.sha256()
    for t in cache_tensors(sequence):
        h.update(str((tuple(t.shape),str(t.dtype))).encode())
        h.update(t.detach().cpu().contiguous().numpy())
    return sequence.seen_tokens,sequence.next_rope_position,h.hexdigest()


def cache_bytes(sequence):return sum(t.numel()*t.element_size() for t in cache_tensors(sequence))


@torch.inference_mode()
def emit_decode(branch,logits,processor,limit,stops,callback):
    generated=[];reason='budget'
    for i in range(limit):
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite output')
        token=int(logits[0,-1].argmax());generated.append(token)
        callback(dict(token_index=i,token_id=token,cumulative_text=processor.tokenizer.decode(generated,skip_special_tokens=True),
            monotonic_s=time.monotonic(),is_stop=token in stops))
        if token in stops:reason='eos';break
        if i+1<limit:logits=branch.append(input_ids=torch.tensor([[token]],device=logits.device),logits_mode='last')
    return dict(text=processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated,stop_reason=reason)


class Session:
    def __init__(self,env,model,builder,*,mode='bounded',arms=None):
        if mode!='bounded' or model.training or next(model.parameters()).dtype!=torch.float32:
            raise ValueError('explicit FP32 mode and eval model required')
        self.env,self.model,self.builder,self.mode=env,model,builder,mode
        self.grid=tuple(n//32 for n in builder.canvas)
        self.pairs=0;self.last_id=self.last_pts=None;self.invalid=self.tail=False
        self.timing=dict(preprocess_seconds=0.,vision_seconds=0.,bank_or_main_update_seconds=0.)
        if mode=='bounded':self.bank=Banks(self.grid,arms=arms)
        else:self.main=env['HybridSequence'](model,max_tokens=262144,session_id='S-EMBER-full-history')

    def sync(self):
        device=self.model.get_input_embeddings().weight.device
        if device.type=='cuda':torch.cuda.synchronize(device)

    @torch.inference_mode()
    def append_pair(self,first,second,*,padding=False):
        if self.invalid or self.tail:raise RuntimeError('closed, failed or finalized-tail session')
        if (self.last_id is not None and (first.frame_id<=self.last_id or first.timestamp<=self.last_pts)
            or (not padding and (first.frame_id>=second.frame_id or first.timestamp>=second.timestamp))
            or (padding and (first.frame_id!=second.frame_id or first.timestamp!=second.timestamp))):
            raise ValueError('ordered real pair or explicit identical EOF pad required')
        prefix=self.builder.prefix_ids.shape[1] if self.mode=='reference' and self.main.seen_tokens==0 else 0
        seen=(self.main.seen_tokens+prefix) if self.mode=='reference' else 0
        rope=(self.main.next_rope_position+prefix) if self.mode=='reference' else 0
        self.sync();started=time.monotonic()
        inputs,record=self.builder.pair(first,second,seen,rope,synthetic_padding=padding)
        device=self.model.get_input_embeddings().weight.device;inputs=inputs.to(device)
        self.sync();prepared=time.monotonic();saved=self.model.model.rope_deltas
        try:
            with self.env['fp32_reference']():
                vision=self.model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True)
                features=torch.cat(vision.pooler_output,0)
                if features.shape!=(self.grid[0]*self.grid[1],self.model.get_input_embeddings().embedding_dim):raise ValueError('unexpected visual grid')
                self.sync();encoded=time.monotonic()
                if self.mode=='bounded':trace=self.bank.append(features,self.pairs,first.timestamp,second.timestamp)
                else:
                    if self.main.seen_tokens+prefix+inputs.input_ids.shape[1]+528>self.main.max_tokens:
                        raise ValueError('full-history reference context exceeded; no reset/truncation')
                    if prefix:self.main.append(input_ids=self.builder.prefix_ids.to(device),logits_mode='none')
                    emb=self.model.get_input_embeddings()(inputs.input_ids)
                    mask=(inputs.input_ids==self.builder.processor.video_token_id).unsqueeze(-1).expand_as(emb)
                    emb=emb.masked_scatter(mask,features.to(emb))
                    for begin in range(0,emb.shape[1],512):
                        self.main.append(inputs_embeds=emb[:,begin:begin+512],position_ids=inputs.position_ids[:,:,begin:begin+512],logits_mode='none')
                    trace=dict(main_seen_tokens=self.main.seen_tokens,main_KV_GDN_bytes=cache_bytes(self.main))
                self.sync();finished=time.monotonic()
        except BaseException:self.invalid=True;raise
        finally:self.model.model.rope_deltas=saved
        self.timing['preprocess_seconds']+=prepared-started;self.timing['vision_seconds']+=encoded-prepared
        self.timing['bank_or_main_update_seconds']+=finished-encoded
        self.pairs+=1;self.last_id,self.last_pts=second.frame_id,second.timestamp;self.tail=padding
        return dict(pairs=self.pairs,record=record,trace=trace,visual_encodings=1)

    def status(self):
        result=dict(mode=self.mode,pairs=self.pairs,last_id=self.last_id,last_pts=self.last_pts,invalid=self.invalid,tail=self.tail)
        if self.mode=='bounded':result.update(banks=self.bank.status(),no_main_language_cache=True)
        else:result.update(main_seen_tokens=self.main.seen_tokens,main_cache_bytes=cache_bytes(self.main),no_capacity_reset=True)
        return result

    @torch.inference_mode()
    def ask(self,arm,question,callback,*,max_new_tokens=16):
        if self.invalid or not self.pairs or not callable(callback) or not 1<=max_new_tokens<=16:
            raise ValueError('valid evidence, callback and bounded generation required')
        if (self.mode=='bounded' and arm not in self.bank.banks) or (self.mode=='reference' and arm!='R_all_history_stream'):
            raise ValueError('arm does not belong to this process mode')
        audit_started=time.monotonic();before=self.status();saved=self.model.model.rope_deltas;branch=None
        if self.mode=='bounded':
            fingerprint=digest(self.bank.banks[arm].nodes());selector=self.bank.selector_digest()
        else:main_before=signature(self.main)
        self.sync();started=time.monotonic();initial_audit_seconds=started-audit_started;first=[]
        def deliver(item):
            if not first:first.append(time.monotonic()-started)
            callback(item)
        try:
            with self.env['fp32_reference']():
                if self.mode=='bounded':
                    bank=self.bank.banks[arm]
                    emb,pos,ids,meta=context(self.model,self.builder,bank.export(),question,visual_limit=bank.budget,max_new_tokens=max_new_tokens)
                    self.sync();prepared=time.monotonic()
                    branch=self.env['HybridSequence'](self.model,max_tokens=4096,session_id='S-EMBER-bank-QA')
                    for begin in range(0,emb.shape[1],512):
                        end=min(begin+512,emb.shape[1])
                        logits=branch.append(inputs_embeds=emb[:,begin:end],position_ids=pos[:,:,begin:end],logits_mode='last' if end==emb.shape[1] else 'none')
                    ids_sha=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                else:
                    ids=self.builder.question_ids(question)
                    if ids.shape[1]>512 or self.main.seen_tokens+ids.shape[1]+max_new_tokens>262144:raise ValueError('full QA suffix/context exceeds limit')
                    branch=self.main.fork('S-EMBER-reference-QA');self.sync();prepared=time.monotonic()
                    logits=branch.append(input_ids=ids.to(self.model.get_input_embeddings().weight.device),logits_mode='last')
                    ids_sha=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                    meta=dict(input_tokens=branch.seen_tokens,visual_tokens=self.pairs*math_prod(self.grid),suffix_tokens=ids.shape[1])
                self.sync();prefilled=time.monotonic()
                eos=self.model.generation_config.eos_token_id
                stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(self.builder.processor.tokenizer.eos_token_id)
                result=emit_decode(branch,logits,self.builder.processor,max_new_tokens,stops,deliver)
                self.sync();finished=time.monotonic();size=cache_bytes(branch)
        finally:
            if branch is not None:branch.cache.reset()
            self.model.model.rope_deltas=saved
        audit_started=time.monotonic();assert self.status()==before
        if self.mode=='bounded':assert digest(self.bank.banks[arm].nodes())==fingerprint and self.bank.selector_digest()==selector
        else:assert signature(self.main)==main_before
        suffix_sha=hashlib.sha256(self.builder.question_ids(question).numpy().tobytes()).hexdigest()
        return dict(arm=arm,**result,**meta,question_suffix_sha256=suffix_sha,
            bounded_context_ids_sha256=ids_sha if self.mode=='bounded' else None,
            first_token_seconds=first[0],QA_total_seconds=finished-started,context_and_fork_seconds=prepared-started,
            prefill_seconds=prefilled-prepared,decode_seconds=finished-prefilled,QA_cache_bytes=size,
            read_state_unchanged=True,branch_released=True,
            state_audit_seconds=initial_audit_seconds+time.monotonic()-audit_started,
            state_audit_cost_in_QA_timer=False,not_unmodified_native_backend=self.mode=='reference')

    def close(self):
        if self.mode=='bounded':self.bank.clear()
        else:self.main.reset()
        self.invalid=True


def math_prod(grid):return grid[0]*grid[1]
