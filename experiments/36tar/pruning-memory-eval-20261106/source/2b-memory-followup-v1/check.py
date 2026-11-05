import os, json, hashlib
from followup import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
if os.name!='nt':os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
proc=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval()
builder=make_builder(env,proc)
from resolution_banks import Banks
from resolution_nodes import digest
bank=Banks((9,12),arms=('LS_mid108','A06_mid108'));control=Banks((9,12),arms=('LS_mid108',))
for i in range(32):
    x=t.randn(108,model.get_input_embeddings().embedding_dim)
    bank.append(x,i,2.*i,2.*i+1);control.append(x,i,2.*i,2.*i+1)
    assert bank.fingerprint()['LS_mid108']==control.fingerprint()['LS_mid108']
h,r=split(bank.banks['LS_mid108']);legacy=legacy_nodes(r);before=bank.fingerprint()
manifest=read(OUT/'manifest.json');layouts=[]
with t.inference_mode():
    for q in manifest['questions']:
        for name,nodes in [('recent',r),('joint',h+r),('A06',bank.export('A06_mid108'))]:
            emb,pos,ids,meta=context(model,builder,nodes,prompt(q),max_new_tokens=16)
            layouts.append((q['id'],name,ids.shape[1],hashlib.sha256(ids.numpy().tobytes()).hexdigest()))
        a=inputs(model,builder,legacy,q['question'],True)
        b=context(model,builder,r,q['question'],visual_limit=432,context_limit=2048,suffix_limit=1024,max_new_tokens=8)
        assert all(t.equal(x,y) for x,y in zip(a[:3],b[:3])),q['id']
    f,_=extract(env,model,builder,legacy,manifest['questions'][0]['question'],True)
    assert f.numel()==model.get_input_embeddings().embedding_dim
    session=Session(env,model,builder,arms=('LS_mid108','A06_mid108'));session.bank=bank;session.pairs=32
    for arm in ('LS_mid108','A06_mid108'):
        z=session.ask(arm,prompt(manifest['questions'][0]),lambda _:None,max_new_tokens=2)
        assert len(z['generated_ids'])<=2
    z=protected_readout.answer(env,model,builder,[],r,prompt(manifest['questions'][0]),isolated=False,limit=2)
    assert bank.fingerprint()==before
    # Short prefixes must bypass the 4-pair extractor without calling the head.
    for n in (1,2,3,4):assert choose(env,model,builder,[],r[:n],'probe',{})['head_called'] is False
    session.close()
result=dict(passed=True,layouts=len(layouts),layout_sha256=hashlib.sha256(json.dumps(layouts).encode()).hexdigest(),
    feature_layouts_equal=150,legacy_type_conversion_exact=True,LS32_updates_unchanged=True,
    tiny_answers=3,tiny_feature_extractions=1,short_prefix_checks=4,real_weights=0,GPU=0)
save(OUT/('CPU_local.json' if os.name=='nt' else 'CPU_cluster.json'),result);print(json.dumps(result))
