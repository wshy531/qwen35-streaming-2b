"""Frozen, explicitly authorized GPU execution; no automatic submission or retry."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
from common import ROOT, PREP, CASES, MATCH, read, write, verify, worker, run_case

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    assert out.resolve().is_relative_to((ROOT/'runs').resolve())
    verify(media=True)
    for side in ('local','remote'):
        assert read(PREP/f'CPU_{side}.json')['status']=='passed_random_weights_only'
    # Permission receipt must bind this exact frozen plan and source manifest.
    from common import sha
    auth=read(PREP/'authorization.json')
    assert auth['approved'] is True
    assert auth['plan_sha256']==sha(PREP/'plan.json') and auth['lock_sha256']==sha(PREP/'execution_lock.json')
    plan=read(PREP/'plan.json');panel={q['id']:q for q in read(PREP/'panel.json')}
    refs={(r['id'],r['arm']):r for r in read(PREP/'reference.json')}
    out.mkdir(exist_ok=False);start=time.monotonic();count=0;tokens=0;calls=0
    def guard():
        if time.monotonic()-start>plan['internal_minutes']*60:
            raise TimeoutError('Predeclared internal stop; no automatic continuation')
    try:
        env=worker.scale.s.runtime();t=env['torch'];t.set_num_threads(1)
        assert t.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        tick=time.perf_counter()
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            ROOT/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,device_map='cuda',
            attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False)
        assert (model.config.vision_config.depth,model.config.text_config.hidden_size)==(24,2048)
        versions={n:p._version for n,p in model.named_parameters()}
        proc=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
        builder=worker.scale.s.make_builder(env,proc);worker.sync(env)
        write(out/'startup.json',dict(model_loads=1,load_seconds=time.perf_counter()-tick,
              versions=dict(torch=t.__version__,transformers=__import__('transformers').__version__,av=__import__('av').__version__),
              gpu=t.cuda.get_device_name(),plan=plan))
        with t.no_grad(),env['fp32_reference']():
            warms=[]
            for case in CASES:
                guard();r=run_case(env,model,builder,read(PREP/'warmup.json'),case,guard)
                warms.append(r);tokens+=len(r['generated_ids']);calls+=r['vision_calls']
            assert calls<=plan['warmup_visual_call_cap']
            write(out/'warmup.json',warms)
            with (out/'answers.jsonl').open('x',encoding='utf-8') as f:
                for job in read(PREP/'jobs.json'):
                    guard();gc.collect();t.cuda.empty_cache();t.cuda.reset_peak_memory_stats()
                    rss_before=env['psutil'].Process().memory_info().rss
                    r=run_case(env,model,builder,panel[job['id']],job['case'],guard)
                    r.update(repeat=job['repeat'],order_position=job['order_position'],
                        peak_allocated_bytes=t.cuda.max_memory_allocated(),peak_reserved_bytes=t.cuda.max_memory_reserved(),
                        host_RSS_before_bytes=rss_before,host_RSS_end_bytes=env['psutil'].Process().memory_info().rss)
                    ref=refs[r['id'],r['arm']]
                    r['reference_match']={k:r[k]==ref[k] for k in MATCH}
                    f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                    count+=1;tokens+=len(r['generated_ids']);calls+=r['vision_calls']
                    assert all(r['reference_match'].values()),('frozen endpoint mismatch',r['id'],r['case'])
                    assert tokens<=plan['maximum_generated_tokens'] and calls<=plan['maximum_visual_calls']
                    print(json.dumps(dict(answers=count,total=96,seconds=time.monotonic()-start)),flush=True)
        assert count==96 and calls==plan['formal_visual_calls']+sum(r['vision_calls'] for r in warms)
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        write(out/'completed.json',dict(status='complete',answers=count,warmup_answers=4,tokens=tokens,
              vision_calls=calls,seconds=time.monotonic()-start,updates=0,base_unchanged=True))
    except BaseException as e:
        write(out/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),answers=count,
              tokens=tokens,vision_calls=calls,seconds=time.monotonic()-start))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
