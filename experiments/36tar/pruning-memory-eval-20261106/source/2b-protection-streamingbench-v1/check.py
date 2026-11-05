import os
from common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
proc=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval()
builder=make_builder(env,proc)
from resolution_banks import Banks
bank=Banks((9,12),arms=('LS_mid108',))
for i in range(12):bank.append(t.randn(108,model.get_input_embeddings().embedding_dim),i,float(2*i),float(2*i+1))
h,r=split(bank.banks['LS_mid108']);layouts=[]
with t.inference_mode():
    for q in read(P/'questions.json'):
        for arm in ARMS:
            _,pos,ids,meta=context(model,builder,r if arm=='recent' else h+r,prompt(q),max_new_tokens=16)
            assert ids.shape[1]+16<=4096
            layouts.append((q['id'],arm,ids.shape[1],hashlib.sha256(ids.numpy().tobytes()).hexdigest()))
    for i in range(0,len(layouts),3):assert layouts[i+1][-1]==layouts[i+2][-1]
    q=read(P/'questions.json')[0];outputs={}
    for arm in ARMS:outputs[arm]=protected_readout.answer(env,model,builder,[] if arm=='recent' else h,r,prompt(q),isolated=arm=='isolated',limit=2)
    assert t.equal(outputs['joint']['position_ids'],outputs['isolated']['position_ids'])
    session=Session(env,model,builder,arms=('LS_mid108',));session.bank=bank;session.pairs=12
    old=session.ask('LS_mid108',prompt(q),lambda _:None,max_new_tokens=2)
    assert old['generated_ids']==outputs['joint']['generated_ids'];session.close()
result=dict(passed=True,synthetic_features=True,layouts=len(layouts),layout_sha256=hashlib.sha256(json.dumps(layouts).encode()).hexdigest(),max_context=max(x[2] for x in layouts),tiny_answers=4,real_weights=0,GPU=0)
save(P/('CPU_local.json' if os.name=='nt' else 'CPU_cluster.json'),result)
print(json.dumps(result),flush=True)
