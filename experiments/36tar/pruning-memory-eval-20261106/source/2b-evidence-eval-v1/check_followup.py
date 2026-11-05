import os,json
from prompt_followup import old,followup_question
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=old.runtime();t=env['torch'];t.set_num_threads(1)
processor=env['AutoProcessor'].from_pretrained(old.R/'models/Qwen3.5-2B',local_files_only=True)
cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
builder=old.make_builder(env,processor);audit={x['video_id']:x for x in old.read(old.DATA/'CPU_media_audit.json')['rows']}
layouts=0;maximum=0
for q in old.read(old.P/'profile.json'):
    samples=audit[q['video_id']]['samples']
    for n in (108,54):
        ids=t.arange(108) if n==108 else t.arange(1,108,2);rec=[]
        for i in range(0,len(samples),2):rec.append((i//2,samples[i]['pts_s'],samples[min(i+1,len(samples)-1)]['pts_s'],t.zeros(n,model.config.text_config.hidden_size),ids))
        _,_,tokens=old.context(model,builder,rec,followup_question(q,0,'B'));assert tokens.shape[1]+128<=4096
        maximum=max(maximum,tokens.shape[1]);layouts+=1
old.save(old.P/'CPU_followup.json',dict(layouts=layouts,max_input=maximum,GPU=0,API=0));print(layouts,maximum)
