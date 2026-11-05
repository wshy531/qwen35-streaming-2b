import sys,json,os
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-prune-timing-audit-v1'))
from audit_common import runtime
from anchor_merge import merge_groups
from protected_cache import compose

def fingerprint(seq):
    import hashlib
    h=hashlib.sha256()
    for field in ('key_cache','value_cache','conv_states','recurrent_states'):
        for x in getattr(seq.cache,field):
            if x is not None:h.update(x.cpu().numpy().tobytes())
    return h.hexdigest(),seq.seen_tokens,seq.next_rope_position

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
x=t.randn(432,24);ids=t.arange(1,108,2)
y,meta=merge_groups(x,ids)
assert y.shape==(216,24) and int(meta['mass'].sum())==108
assert t.allclose((y.reshape(54,4,-1)*meta['mass'][:,None,None]).sum(0),x.reshape(108,4,-1).sum(0),atol=1e-5)
identity,_=merge_groups(x,t.arange(108));assert t.equal(identity,x)
constant=t.ones_like(x);z,_=merge_groups(constant,ids);assert t.equal(z,t.ones_like(z))
processor=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
cfg=env['tiny_config'](processor)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
model.requires_grad_(False)
prefix=t.tensor([[10,11,12,13]]);new=t.tensor([[20,21,22,23]])
with t.inference_mode(),env['fp32_reference']():
    recent=env['HybridSequence'](model,max_tokens=64);recent.append(input_ids=prefix)
    pos=t.stack((t.arange(4,8),t.arange(8,12),t.arange(8,12),t.arange(8,12)))[:,None,:]
    recent.append(input_ids=new,position_ids=pos)
    saved=fingerprint(recent);branches=[]
    for values in ([30,31,32,33],[40,41,42,43]):
        hist=env['HybridSequence'](model,max_tokens=64);hist.append(input_ids=prefix);hist.append(input_ids=t.tensor([values]))
        before=fingerprint(hist);branch=compose(hist,recent,4)
        assert branch.seen_tokens==12
        for field in ('conv_states','recurrent_states'):
            for a,b in zip(getattr(branch.cache,field),getattr(recent.cache,field)):
                if a is not None:assert t.equal(a,b) and a.data_ptr()!=b.data_ptr()
        logits=branch.append(input_ids=t.tensor([[50,51]]),logits_mode='last')
        assert t.isfinite(logits).all() and fingerprint(recent)==saved and fingerprint(hist)==before
        branches.append(branch);hist.cache.reset()
    empty=env['HybridSequence'](model,max_tokens=64);empty.append(input_ids=prefix)
    copied=compose(empty,recent,4);assert fingerprint(copied)==saved
    for seq in branches+[copied,empty,recent]:seq.cache.reset()
out=R/'runs/protection_prototypes_20261103_01';out.mkdir(exist_ok=False)
(out/'CPU.json').write_text(json.dumps(dict(status='passed',merge_shape=[216,24],mass_conserved=True,
    all_keep_exact=True,constant_exact=True,recent_states_clone_exact=True,history_swap_does_not_mutate_recent=True,
    empty_history_exact=True,tiny_query_continuations=2,GPU=0,pretrained_loads=0,ability_claim=False),indent=2))
print('Local merge contracts and two tiny hybrid splice continuations passed; NO capability evidence')
