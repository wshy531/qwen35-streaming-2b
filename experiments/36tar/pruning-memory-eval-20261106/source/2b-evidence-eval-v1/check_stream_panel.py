import hashlib,os,json
import gpu_profile as s
from response import question
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=s.runtime();t=env['torch'];t.set_num_threads(1)
processor=env['AutoProcessor'].from_pretrained(s.R/'models/Qwen3.5-2B',local_files_only=True)
cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
builder=s.make_builder(env,processor);qs=[q for q in s.read(s.P/'panel_media_bound.json') if q['dataset']=='StreamingBench']
layouts=0;maximum=0;digest=hashlib.sha256()
for q in qs:
    # Conservative complete-pair bound for layout only; actual frames verified by causal driver.
    for count in (108,54):
        n=min(8,(int(q['query_time_s'])+1)//2);assert n>0
        ids=t.arange(108) if count==108 else t.arange(1,108,2)
        rec=[(i,float(2*i),float(2*i+1),t.zeros(count,model.config.text_config.hidden_size),ids) for i in range(n)]
        for shift in (0,2):
            for proto in ('A','B'):
                _,pos,tokens=s.context(model,builder,rec,question(q,shift,proto));assert tokens.shape[1]+(8 if proto=='A' else 128)<=4096
                layouts+=1;maximum=max(maximum,tokens.shape[1]);digest.update(tokens.numpy().tobytes());digest.update(pos.numpy().tobytes())
budget=3*sum((int(max(q['query_time_s'] for q in qs if q['source_id']==v))+2)//2 for v in {q['source_id'] for q in qs})
out=dict(layouts=layouts,max_input=maximum,layout_sha256=digest.hexdigest(),nominal_vision_bound=budget,
    actual_PTS_still_runtime_checked=True,GPU=0,API=0)
s.save(s.P/'CPU_stream_panel.json',out);print(json.dumps(out))
