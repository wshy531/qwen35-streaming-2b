import os,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-isolated-recent-v1'))
from experiment import runtime,make_builder,context,split
from protected_cache import compose
from channel_intervention import cross_channels
from resolution_banks import Banks
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
proc=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval()
b=make_builder(env,proc);bank=Banks((9,12),arms=('LS_mid108',))
for i in range(12):bank.append(t.randn(108,model.get_input_embeddings().embedding_dim),i,2*i,2*i+1)
h,r=split(bank.banks['LS_mid108'])
def fingerprint(s):
    d=hashlib.sha256()
    for field in ('key_cache','value_cache','conv_states','recurrent_states'):
        for x in getattr(s.cache,field):
            if x is not None:d.update(x.cpu().numpy().tobytes())
    return d.hexdigest()
with t.inference_mode(),env['fp32_reference']():
    emb,pos,ids,meta=context(model,b,h+r,'What is happening now? A. Open B. Close',max_new_tokens=2)
    _,_,hids,hm=context(model,b,h,'What is happening now? A. Open B. Close',max_new_tokens=2)
    prefix=b.prefix_ids.shape[1];end=emb.shape[1]-meta['suffix_tokens'];boundary=hids.shape[1]-hm['suffix_tokens']
    base=env['HybridSequence'](model,max_tokens=4096);base.append(inputs_embeds=emb[:,:prefix],position_ids=pos[:,:,:prefix],logits_mode='none')
    joint=base.fork('joint');hist=base.fork('hist');recent=base.fork('recent')
    joint.append(inputs_embeds=emb[:,prefix:end],position_ids=pos[:,:,prefix:end],logits_mode='none')
    hist.append(inputs_embeds=emb[:,prefix:boundary],position_ids=pos[:,:,prefix:boundary],logits_mode='none')
    p=pos[:,:,boundary:end].clone();p[0,0]=t.arange(prefix,prefix+end-boundary)
    recent.append(inputs_embeds=emb[:,boundary:end],position_ids=p,logits_mode='none')
    isolated=compose(hist,recent,prefix);before=(fingerprint(joint),fingerprint(isolated))
    combinations=[]
    for a,l in [(joint,joint),(joint,isolated),(isolated,joint),(isolated,isolated)]:
        c=cross_channels(a,l)
        for f in ('key_cache','value_cache'):
            assert all(x is None or t.equal(x,y) for x,y in zip(getattr(c.cache,f),getattr(a.cache,f)))
        for f in ('conv_states','recurrent_states'):
            assert all(x is None or t.equal(x,y) for x,y in zip(getattr(c.cache,f),getattr(l.cache,f)))
        if a is l:assert fingerprint(c)==fingerprint(a)
        z=c.append(inputs_embeds=emb[:,end:],position_ids=pos[:,:,end:],logits_mode='last')
        assert t.isfinite(z).all();combinations.append(int(z.argmax(-1)));c.cache.reset()
    assert before==(fingerprint(joint),fingerprint(isolated))
    for s in (base,joint,hist,recent,isolated):s.cache.reset()
out=R/'runs/protection_channel_mechanism_20261103_01';out.mkdir(exist_ok=False)
(out/'CPU.json').write_text(json.dumps(dict(passed=True,combinations=4,finite_first_logits=True,source_unchanged=True,diagonal_cache_exact=True,real_weights=0,GPU=0,caveat='State-bundle intervention only, no claim of video benefit'),indent=2),encoding='utf-8')
print('four channel combinations passed; random model only')
