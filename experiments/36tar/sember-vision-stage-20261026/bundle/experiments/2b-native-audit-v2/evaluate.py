"""Approved-only original API profile. No model patch, continuation wrapper, or training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from audit_common import ROOT,OUT,MODEL,read,save,official_identity,tensor_sha
from native_inputs import matched,normal_video


def verify():
    lock=read(OUT/'inputs/execution_lock.json')
    for name,digest in lock['runtime_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in lock['input_sha256'].items():
        assert hashlib.sha256((OUT/'inputs'/name).read_bytes()).hexdigest()==digest,name
    return lock


def main(condition,out,deadline):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('new explicit GPU approval and actual Slurm allocation required')
    lock=verify()
    import numpy as np
    import psutil
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
    torch.set_num_threads(1)
    version='5.3.0' if condition=='B0_official530_matched_pairs' else '5.4.0'
    assert official_identity()['transformers_version']==version
    source_key='original_official_source_SHA' if version=='5.3.0' else 'new_official_source_SHA'
    assert official_identity()['model_source_sha256']==lock[source_key]
    assert torch.cuda.device_count()==1
    assert torch.cuda.mem_get_info()[0]>=32*2**30
    assert out.resolve().is_relative_to((ROOT/'runs').resolve())
    out.mkdir(exist_ok=False)
    def guard():
        if time.time()>=deadline:raise TimeoutError('fixed profile deadline; no automatic continuation')
        assert psutil.Process().memory_info().rss<20*2**30,'RSS budget exceeded'
        assert torch.cuda.max_memory_allocated()<48*2**30,'allocated GPU budget exceeded'
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())<128*2**20
    started=time.monotonic()
    model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    model.eval()
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys'),loading
    identity=official_identity(model)
    load_seconds=time.monotonic()-started
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    vision_calls=[]
    model.model.visual.register_forward_pre_hook(lambda module,args:vision_calls.append(1))
    # One fixed synthetic pair for warmup, never a formal question or retained history.
    from audit_common import paired_inputs
    frames=[np.zeros((64,96,3),dtype=np.uint8),np.full((64,96,3),32,dtype=np.uint8)]
    warm=paired_inputs(processor,frames,[0.,2.],'What is visible?').to('cuda')
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        generated=model.generate(**warm,do_sample=False,max_new_tokens=16,logits_to_keep=1,
            use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
    warm_tokens=generated.shape[1]-warm.input_ids.shape[1]
    assert len(vision_calls)==1
    del warm,generated
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    save(out/'startup.json',dict(condition=condition,identity=identity,load_seconds=load_seconds,
        loading_info=loading,warmup_answers=1,warmup_generated_tokens=warm_tokens,pretrained_loads=1,
        FP32=True,attention='official SDPA with PyTorch MATH backend, same as current C',
        original_generate=True,project_model_patches=0))
    questions=read(OUT/'inputs/requests.json');expected=read(OUT/'inputs/expected_prefixes.json')
    video_meta={v['video_id']:v for v in read(OUT/'inputs/videos.json')}
    data=ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23'
    total=0;began=time.monotonic()
    with (out/'answers.jsonl').open('x',encoding='utf-8') as stream:
        for serial,q in enumerate(questions,1):
            guard();sample_started=time.monotonic()
            v=video_meta[q['video_id']];path=(data/v['relative_path']).resolve()
            assert path.is_relative_to(data.resolve()) and path.stat().st_size==v['bytes']
            inputs,meta=normal_video(path,q) if condition.startswith('A_') else matched(path,q,expected[q['question_id']])
            assert inputs.input_ids.shape[1]+16<=262144
            guard();prepared=time.monotonic()
            inputs=inputs.to('cuda');model.model.rope_deltas=None
            torch.cuda.synchronize();qa_started=time.monotonic()
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                outputs=model.generate(**inputs,do_sample=False,max_new_tokens=16,use_cache=True,
                    logits_to_keep=1,return_dict_in_generate=True,output_scores=True,
                    pad_token_id=processor.tokenizer.eos_token_id)
            torch.cuda.synchronize();finished=time.monotonic()
            ids=outputs.sequences[0,inputs.input_ids.shape[1]:].tolist()
            assert len(vision_calls)==1+serial
            first=outputs.scores[0].detach().float().cpu()
            assert len(ids)<=16 and torch.isfinite(first).all()
            name=q['question_id']+'.first_logits.npy';np.save(out/name,first.numpy())
            row=dict(question_id=q['question_id'],video_id=q['video_id'],condition=condition,
                generated_ids=ids,text=processor.tokenizer.decode(ids,skip_special_tokens=True),input=meta,
                first_logits_file=name,first_logits_sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),
                first_argmax=int(first.argmax()),first_margin=float(first.topk(2).values.diff().abs().item()),
                input_seconds=prepared-sample_started,QA_seconds=finished-qa_started,
                sample_wall_seconds=finished-sample_started,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();total+=len(ids)
            del outputs,inputs,first
            gc.collect();torch.cuda.empty_cache();guard()
            assert official_identity(model)==identity
    save(out/'completed.json',dict(status='complete',condition=condition,answers=len(questions),
        generated_tokens=total,warmup_answers=1,warmup_generated_tokens=warm_tokens,
        pretrained_loads=1,wall_seconds=time.monotonic()-began,
        vision_encodings=len(vision_calls)-1,warmup_encodings=1,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),last_RSS_bytes=psutil.Process().memory_info().rss))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=[
        'A_official540_default_video','B_official540_matched_pairs','B0_official530_matched_pairs'],required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();main(args.condition,args.out,args.deadline)
