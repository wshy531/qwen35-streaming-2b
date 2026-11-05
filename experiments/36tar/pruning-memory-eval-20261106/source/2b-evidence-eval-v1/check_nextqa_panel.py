import os,json,hashlib
import gpu_profile as s
from response import question
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=s.runtime();t=env['torch'];t.set_num_threads(1)
processor=env['AutoProcessor'].from_pretrained(s.R/'models/Qwen3.5-2B',local_files_only=True)
cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
builder=s.make_builder(env,processor);audit={r['video_id']:r for r in s.read(s.DATA/'CPU_media_audit.json')['rows']}
qs=[q for q in s.read(s.P/'panel.json') if q['dataset']=='NExT-QA'];layouts=0;maximum=0;digest=hashlib.sha256()
for q in qs:
    samples=audit[q['video_id']]['samples']
    for count in (108,54):
        ids=t.arange(108) if count==108 else t.arange(1,108,2);rec=[]
        for i in range(0,len(samples),2):rec.append((i//2,samples[i]['pts_s'],samples[min(i+1,len(samples)-1)]['pts_s'],t.zeros(count,model.config.text_config.hidden_size),ids))
        for shift in (0,2):
            for proto in ('A','B'):
                _,pos,tokens=s.context(model,builder,rec,question(q,shift,proto))
                assert tokens.shape[1]+(8 if proto=='A' else 128)<=4096
                maximum=max(maximum,tokens.shape[1]);layouts+=1;digest.update(tokens.numpy().tobytes());digest.update(pos.numpy().tobytes())
vids={q['source_id'] for q in qs};vision=3*sum((len(audit[v]['samples'])+1)//2 for v in vids)
s.save(s.P/'CPU_nextqa_panel.json',dict(layouts=layouts,max_tokens=maximum,layout_sha256=digest.hexdigest(),vision_calls=vision,GPU=0,API=0))
print(layouts,maximum,vision)
