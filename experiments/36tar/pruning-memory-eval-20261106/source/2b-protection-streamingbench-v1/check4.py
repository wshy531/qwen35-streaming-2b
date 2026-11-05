import os
from fair_common import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
proc=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval();builder=make_builder(env,proc)
from resolution_banks import Banks
bank=Banks((9,12),arms=('LS_mid108','M0_mid108'));old=Banks((9,12),arms=('LS_mid108',))
for i in range(12):
    x=t.randn(108,model.get_input_embeddings().embedding_dim)
    bank.append(x,i,float(2*i),float(2*i+1));old.append(x,i,float(2*i),float(2*i+1))
    assert bank.fingerprint()['LS_mid108']==old.fingerprint()['LS_mid108']
h,r=split(bank.banks['LS_mid108']);m0=bank.export('M0_mid108');assert len(m0)==8
layouts=[]
with t.inference_mode():
    for q in read(P/'questions.json'):
        for arm in ARMS:
            nodes=r if arm=='recent' else m0 if arm=='recent864' else h+r
            _,pos,ids,meta=context(model,builder,nodes,prompt(q),max_new_tokens=16)
            assert ids.shape[1]+16<=4096 and meta['visual_tokens']==(432 if arm=='recent' else 864)
            layouts.append((q['id'],arm,ids.shape[1],hashlib.sha256(ids.numpy().tobytes()).hexdigest()))
    previous=[x for x in layouts if x[1]!='recent864']
    assert hashlib.sha256(json.dumps(previous).encode()).hexdigest()==read(P/'CPU_local.json')['layout_sha256']
    txt=prompt(read(P/'questions.json')[0]);session=Session(env,model,builder,arms=('LS_mid108','M0_mid108'));session.bank=bank;session.pairs=12
    answers={}
    for arm in ARMS:
        if arm in ('joint','recent864'):z=session.ask('LS_mid108' if arm=='joint' else 'M0_mid108',txt,lambda _:None,max_new_tokens=2)
        else:z=protected_readout.answer(env,model,builder,h if arm=='isolated' else [],r,txt,isolated=arm=='isolated',limit=2)
        answers[arm]=z
    control=protected_readout.answer(env,model,builder,[],m0,txt,isolated=False,limit=2)
    assert control['generated_ids']==answers['recent864']['generated_ids']
    session.close()
result=dict(passed=True,synthetic_features=True,layouts=len(layouts),layout_sha256=hashlib.sha256(json.dumps(layouts).encode()).hexdigest(),old_450_layouts_exact=True,LS_12_updates_exact_with_added_M0=True,tiny_answers=5,real_weights=0,GPU=0,max_context=max(x[2] for x in layouts))
save(P/('CPU4_local.json' if os.name=='nt' else 'CPU4_cluster.json'),result);print(json.dumps(result),flush=True)
