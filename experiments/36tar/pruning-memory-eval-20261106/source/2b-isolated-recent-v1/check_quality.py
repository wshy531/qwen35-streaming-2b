"""CPU bounds and actual tiny-model readout; synthetic features, no real weights."""
import os
from experiment import *
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
proc=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
print('processor ready',flush=True)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval()
builder=make_builder(env,proc)
from resolution_banks import Banks
bank=Banks((9,12),arms=('LS_mid108',))
for i in range(12):bank.append(t.randn(108,model.get_input_embeddings().embedding_dim),i,float(4*i),float(4*i+2))
h,r=split(bank.banks['LS_mid108']);assert len(h)==len(r)==4
print('tiny model and bank ready',flush=True)
tasks=[(q['question_id'],0,prompt(q)) for q in read(P/'requests.json')]
tasks += [(q['id'],s,question_prompt(rotate(q,s)[0])) for q in read(P/'paired_questions.json') for s in range(5)]
layouts=[]
with t.inference_mode():
    for key,s,txt in tasks:
        for arm in ARMS:
            _,pos,ids,meta=context(model,builder,r if arm=='recent' else h+r,txt,max_new_tokens=16)
            assert ids.shape[1]+16<=4096 and meta['visual_tokens']==(432 if arm=='recent' else 864)
            layouts.append((key,s,arm,ids.shape[1],sha(P/'protocol.json') if False else hashlib.sha256(ids.numpy().tobytes()).hexdigest()))
    for i in range(0,len(layouts),3):assert layouts[i+1][-1]==layouts[i+2][-1]
    print('layouts',len(layouts),flush=True)
    txt=tasks[0][2]
    a=protected_readout.answer(env,model,builder,h,r,txt,isolated=True,limit=2)
    print('isolated tiny answer ready',flush=True)
    b=protected_readout.answer(env,model,builder,h,r,txt,isolated=False,limit=2)
    c=protected_readout.answer(env,model,builder,[],r,txt,isolated=True,limit=2)
    assert t.equal(a['input_ids'],b['input_ids']) and t.equal(a['position_ids'],b['position_ids'])
    assert a['read_state_unchanged'] and c['branch_released']
    # Same old Session readout as the GPU joint comparator.
    session=Session(env,model,builder,arms=('LS_mid108',));session.bank=bank;session.pairs=12
    d=session.ask('LS_mid108',txt,lambda _:None,max_new_tokens=2)
    assert b['generated_ids']==d['generated_ids']
loaded=sorted({Path(m.__file__).resolve().relative_to(R.resolve()).as_posix() for m in list(sys.modules.values())
    if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(R.resolve())
    and ('experiments' in Path(m.__file__).parts or 'repo' in Path(m.__file__).parts or 'publish' in Path(m.__file__).parts)})
result=dict(passed=True,layouts=len(layouts),layout_sha256=hashlib.sha256(json.dumps(layouts).encode()).hexdigest(),
    max_context=max(x[3] for x in layouts),tiny_answers=4,pretrained_loads=0,GPU=0,
    synthetic_visual_features=True,actual_media_not_read=True,joint_isolated_same_ids_positions=True,
    old_session_tiny_joint_answer_exact=True)
save(P/('CPU_local.json' if os.name=='nt' else 'CPU_cluster.json'),result)
save(P/('loaded_files.json' if os.name=='nt' else 'loaded_files_cluster.json'),loaded)
print(json.dumps(result))

