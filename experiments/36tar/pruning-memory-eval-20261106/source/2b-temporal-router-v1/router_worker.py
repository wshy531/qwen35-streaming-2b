"""Separately authorized GPU task: classify question text, no video and no QA."""
import argparse
import gc
import os
from pathlib import Path
import time
from route_common import ROOT,INPUTS,read,save,sha
from temporal_policy import STAGES,render,route_question
from route_runtime import runtime,score_view


def main(out,lock_path):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':
        raise RuntimeError('approved one-GPU/four-CPU Slurm allocation required')
    if not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('owned result path required')
    from freeze_router import verify
    lock=verify(lock_path)
    # No branch outputs, evaluator labels or task metadata are loaded here.
    requests=read(INPUTS/'router_requests.json')
    if len(requests)!=113 or any(set(r)!={'id','question'} for r in requests):
        raise ValueError('frozen question-only requests required')
    env=runtime();torch=env['torch'];psutil=env['psutil'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<24*2**30:
        raise RuntimeError('one GPU and declared memory margin required')
    if psutil.virtual_memory().available<12*2**30:raise RuntimeError('system memory margin below 12 GiB')
    out.mkdir(exist_ok=False);started=time.monotonic();calls=[0];visual=[0]
    def guard():
        if time.monotonic()-started>720 or psutil.Process().memory_info().rss>20*2**30:
            raise RuntimeError('worker time/RSS bound exceeded')
        if torch.cuda.max_memory_allocated()>48*2**30:raise RuntimeError('GPU-memory bound exceeded')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>32*2**20:
            raise RuntimeError('result-size bound exceeded')
    model,loading=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('loading mismatch')
    if model.config.text_config.hidden_size!=2048:raise ValueError('frozen 2B width required')
    model.eval()
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    tokenizer=processor.tokenizer
    hook=model.model.visual.register_forward_pre_hook(lambda *args:visual.__setitem__(0,visual[0]+1))
    save(out/'startup.json',dict(pretrained_loads=1,precision='float32',lock_sha256=sha(lock_path),
        loading_summary={k:len(loading.get(k) or ()) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')},
        labels_or_branch_predictions_read=False))
    warm=[]
    def scored(prompt):
        guard();value=score_view(env,model,tokenizer,prompt);calls[0]+=1;return value
    for stage,states in STAGES.items():
        for shift in range(len(states)):
            value=scored(render(requests[0]['question'],stage,shift))
            warm.append(dict(stage=stage,shift=shift,**value))
    save(out/'warmup.json',warm)
    gc.collect();torch.cuda.empty_cache()
    records=[]
    with (out/'router_progress.jsonl').open('x',encoding='utf-8') as stream:
        import json
        for i,request in enumerate(requests):
            before=time.monotonic()
            result=route_question(request['question'],scored)
            if i==0:
                a,b=result['support']['views'][0],warm[0]
                if a['input_ids_sha256']!=b['input_ids_sha256'] or not torch.allclose(
                    torch.tensor(a['candidate_logits']),torch.tensor(b['candidate_logits']),rtol=1e-5,atol=1e-6):
                    raise RuntimeError('same-input warm/formal classification changed')
            row=dict(id=request['id'],result=result,total_router_seconds=time.monotonic()-before)
            records.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            if (i+1)%10==0:print(dict(completed_questions=i+1,classifier_forwards=calls[0]),flush=True)
    guard()
    if len(records)!=113 or calls[0]>1026 or visual[0]!=0:raise RuntimeError('work budget drift')
    save(out/'router_predictions.json',records)
    save(out/'completed.json',dict(status='complete',question_texts=113,classifier_forwards=calls[0],
        formal_classifier_forwards=calls[0]-9,warmup_classifier_forwards=9,
        pretrained_loads=1,generated_tokens=0,new_QA_answers=0,visual_forwards=visual[0],
        trained=False,wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),RSS_bytes=psutil.Process().memory_info().rss,
        conditional_stage2=True,automatic_retry=False))
    hook.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    a=p.parse_args();main(a.out,a.lock)
