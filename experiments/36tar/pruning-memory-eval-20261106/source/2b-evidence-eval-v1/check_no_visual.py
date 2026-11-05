import os,json
import no_visual_control as control
s=control.s
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=s.runtime();t=env['torch'];t.set_num_threads(1);s.runtime_torch=t
processor=env['AutoProcessor'].from_pretrained(s.R/'models/Qwen3.5-2B',local_files_only=True)
cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
builder=s.make_builder(env,processor);cases=s.read(s.P/'real_pilot_LOCAL_ONLY.json');seen=set();maximum=0
for case in cases:
    if case['question_id'] in seen:continue
    seen.add(case['question_id']);q=dict(question=case['question'],options=case['options'])
    emb,pos,ids=control.text_context(model,builder,None,control.s.question(q,0,'B'));maximum=max(maximum,len(ids[0]))
    assert pos.shape==(4,1,ids.shape[1]) and ids.shape[1]+128<=4096
s.context=control.text_context
with t.no_grad(),env['fp32_reference']():out=s.generate(env,model,builder,None,q,'B')
s.save(s.P/'CPU_no_visual.json',dict(layouts=len(seen),max_input=maximum,micro_generated=len(out['generated_ids']),visual_tokens=0,GPU=0,API=0))
print(len(seen),maximum)
