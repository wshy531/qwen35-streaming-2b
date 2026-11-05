import os,json,sys
from pathlib import Path
from channel_readout import *
from experiment import runtime,make_builder,split,protected_readout,compact,read,save,question_prompt,rotate
from resolution_banks import Banks
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
P=R/'runs/protection_channels_prepare_20261103_01';P.mkdir(exist_ok=True)
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
proc=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval();builder=make_builder(env,proc)
bank=Banks((9,12),arms=('LS_mid108',))
for i in range(12):bank.append(t.randn(108,model.get_input_embeddings().embedding_dim),i,2*i,2*i+1)
h,r=split(bank.banks['LS_mid108']);text='What is happening now? A. Open B. Close'
with t.inference_mode():
    layouts=[]
    for q in read(R/'runs/isolated_recent_prepare_20261103_01/paired_questions.json'):
        for shift in range(5):
            _,pos,ids,meta=context(model,builder,h+r,question_prompt(rotate(q,shift)[0]),max_new_tokens=16)
            assert ids.shape[1]+16<=4096
            for arm in ARMS:layouts.append((q['id'],shift,arm,ids.shape[1],hashlib.sha256(ids.numpy().tobytes()).hexdigest()))
    answers=answer_four(env,model,builder,h,r,text,limit=2)
    for name in ('joint','isolated'):
        old=compact(protected_readout.answer(env,model,builder,h,r,text,isolated=name=='isolated',limit=2))
        for key in ('generated_ids','text','bounded_context_ids_sha256','position_sha256','visual_tokens'):assert old[key]==answers[name][key],key
    empty=answer_four(env,model,builder,[],r,text,limit=2)
    assert all(x==empty['joint'] for x in empty.values())
result=dict(passed=True,layouts=len(layouts),layout_sha256=hashlib.sha256(json.dumps(layouts).encode()).hexdigest(),tiny_answers=10,real_weights=0,GPU=0,endpoints_match_old=True,empty_history_all_equal=True)
save(P/('CPU_local.json' if os.name=='nt' else 'CPU_cluster.json'),result);print(json.dumps(result))
