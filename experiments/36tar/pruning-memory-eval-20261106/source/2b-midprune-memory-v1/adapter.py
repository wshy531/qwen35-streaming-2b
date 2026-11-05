"""Sparse visual projections into existing bounded memory policies; no padded fake tokens."""
import hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-resolution-cost-v1'))
import torch
from resolution_nodes import Node,validate,digest
from resolution_banks_base import Bank,distance
from resolution_common import THRESHOLD
from resolution_runtime import context
POLICIES=('M0','M2','A06','LS')
VIEWS=('full','fixed12','heuristic11','mask_ce')

def sparse_pair(features,ids,index,begin,end):
    if (features.ndim!=2 or features.shape[0] not in (54,108) or ids.ndim!=1
        or len(ids)!=len(features) or ids.dtype!=torch.long):raise ValueError('54/108 real projections and original IDs required')
    x=features.detach().to(device='cpu',dtype=torch.float32).clone()
    ids=ids.detach().to('cpu').clone()
    if not torch.isfinite(x).all() or bool(torch.any(ids<0)) or bool(torch.any(ids>=108)) or bool(torch.any(ids[1:]<=ids[:-1])):
        raise ValueError('finite features and sorted distinct original 9x12 coordinates required')
    coords=torch.stack((torch.full_like(ids,index),ids//12,ids%12),1)
    n=Node(x,torch.ones((len(x),1)),coords,index,index,float(begin),float(end),(begin+end)/2,grid=(9,12))
    validate(n);return n

class Memory:
    """One visual path per instance. No full projection archive or main language KV."""
    def __init__(self,view):
        if view not in VIEWS:raise ValueError('registered visual path required')
        self.view=view;self.keep=108 if view=='full' else 54
        self.banks={p:Bank(p,self.keep,budget=864,recent=432,closed=324) for p in POLICIES}
        if self.keep==54:self.banks['M0_same8']=Bank('M0',54,budget=432,recent=216,closed=162)
        self.previous_descriptor=None;self.index=0;self.last_end=None;self.closed=False

    @torch.no_grad()
    def append(self,features,ids,index,begin,end):
        if self.closed or index!=self.index or (self.last_end is not None and begin<=self.last_end):raise ValueError('ordered append required')
        if len(features)!=self.keep:raise ValueError('fixed visual path size required')
        node=sparse_pair(features,ids,index,begin,end)
        desc=node.features.mean(0)
        changed=None if self.previous_descriptor is None else distance(self.previous_descriptor,desc)
        starts=changed is None or changed>THRESHOLD
        try:
            for bank in self.banks.values():bank.append(node,starts_event=starts)
        except BaseException:self.closed=True;raise
        self.previous_descriptor=desc.clone();self.index+=1;self.last_end=end
        return dict(pair_index=index,starts_event=starts,adjacent_distance=changed,
            event_descriptor='own projected features, no unpruned reference needed',
            boundaries_may_change_with_pruning=True,selected_original_ids=ids.detach().cpu().tolist())

    def export(self,policy):
        if self.closed:raise RuntimeError('closed memory')
        return self.banks[policy].export()

    def fingerprint(self):
        h=hashlib.sha256()
        for p,b in self.banks.items():h.update(p.encode());h.update(digest(b.nodes()).encode())
        if self.previous_descriptor is not None:h.update(self.previous_descriptor.numpy().tobytes())
        return h.hexdigest()

    def status(self):
        d=0 if self.previous_descriptor is None else self.previous_descriptor.numel()*self.previous_descriptor.element_size()
        return dict(view=self.view,pairs=self.index,last_end=self.last_end,closed=self.closed,
            banks={p:dict(b.status(),visual_budget=b.budget,standalone_descriptor_bytes=0 if p.startswith('M0') else d) for p,b in self.banks.items()},
            shared_descriptor_bytes=d,no_full_history_archive=True,no_main_language_cache=True,
            encoder_previous_state_not_in_this_bank=True)

    def close(self):
        self.banks.clear();self.previous_descriptor=None;self.closed=True

@torch.no_grad()
def ask(env,model,builder,memory,policy,prompt,limit=128):
    if not 1<=limit<=128:raise ValueError('bounded answer required')
    before=memory.fingerprint();saved=model.model.rope_deltas;branch=None
    try:
        with env['fp32_reference']():
            emb,pos,ids,meta=context(model,builder,memory.export(policy),prompt,
                visual_limit=memory.banks[policy].budget,max_new_tokens=min(limit,16))
            if ids.shape[1]+limit>4096:raise ValueError('full output reserve must fit; no truncation')
            branch=env['HybridSequence'](model,max_tokens=4096,session_id='midprune-memory-QA')
            for begin in range(0,emb.shape[1],512):
                end=min(begin+512,emb.shape[1])
                logits=branch.append(inputs_embeds=emb[:,begin:end],position_ids=pos[:,:,begin:end],logits_mode='last' if end==emb.shape[1] else 'none')
            eos=model.generation_config.eos_token_id
            stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
            generated,reason=env['greedy_decode'](branch,logits,limit,stops)
    finally:
        if branch is not None:branch.cache.reset()
        model.model.rope_deltas=saved
    assert memory.fingerprint()==before
    return dict(text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated,
        reason=reason,read_state_unchanged=True,branch_released=True,input_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        position_ids_sha256=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest(),**meta)
