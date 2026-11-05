"""One FP32 model, all 478 complete MCQs, zero vision and fresh QA state."""
import gc
import hashlib
import json
import os
import time
from text_common import ROOT,OUT,read,save,sha,runtime,make_builder,text_ids,verify
from text_inference import answer


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('specific approved GPU allocation required')
    protocol=verify();auth=read(OUT/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_SHA']==sha(OUT/'inputs/execution_lock.json')
    assert read(OUT/'remote_CPU_preflight.json')['status']=='passed'
    assert read(OUT/'CPU_comparison.json')['status']=='passed'
    out=ROOT/'runs'/('2b-text-control-'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    deadline=time.time()+protocol['resource_budget']['minutes']*60-60
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>=16*2**30
    assert env['psutil'].virtual_memory().available>=12*2**30
    started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor);load_seconds=time.monotonic()-started
    def forbid(*args,**kwargs):raise AssertionError('unexpected vision call in text-only control')
    hook=model.model.visual.register_forward_pre_hook(forbid)
    qs=sorted(read(OUT/'inputs/requests.json'),key=lambda q:(q['video_id'],q['question_time'],q['question_id']))
    specs=read(OUT/'inputs/text_specs.json');ids,_=text_ids(builder,qs[0])
    warm=answer(env,model,processor,ids,lambda item:None)
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    save(out/'startup.json',dict(pretrained_loads=1,load_seconds=load_seconds,warmup_answers=1,
        warmup_generated_tokens=len(warm['generated_ids']),vision_calls=0,video_decodes=0))
    began=time.monotonic();generated=0
    with (out/'answers.jsonl').open('x',encoding='utf-8') as rows,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens:
        for i,q in enumerate(qs):
            if time.time()>=deadline:raise TimeoutError('approved walltime reached; no automatic continuation')
            assert env['psutil'].Process().memory_info().rss<protocol['worker_RSS_limit_bytes']
            assert torch.cuda.max_memory_allocated()<protocol['GPU_allocated_limit_bytes']
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<protocol['log_limit_bytes']
            ids,spec=text_ids(builder,q);assert spec==specs[q['question_id']]
            def emit(item):tokens.write(json.dumps(dict(question_id=q['question_id'],**item))+'\n')
            r=answer(env,model,processor,ids,emit)
            rows.write(json.dumps(dict(question_id=q['question_id'],video_id=q['video_id'],arm='text_only',
                input_sha256=spec['input_sha256'],MCQ_text_sha256=spec['MCQ_text_sha256'],
                question_suffix_sha256=spec['question_suffix_sha256'],**r))+'\n')
            rows.flush();tokens.flush();generated+=len(r['generated_ids'])
            if (i+1)%25==0:print(dict(answers=i+1,generated_tokens=generated),flush=True)
    hook.remove()
    assert len(qs)==478 and generated+len(warm['generated_ids'])<=7664
    save(out/'completed.json',dict(status='complete',answers=478,warmup_answers=1,
        generated_tokens=generated,warmup_generated_tokens=len(warm['generated_ids']),pretrained_loads=1,
        vision_calls=0,video_decodes=0,wall_seconds=time.monotonic()-began,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':main()
