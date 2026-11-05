"""Fixed repeatability and same-device recompute diagnostic, zero updates."""
import argparse
import gc
import json
import os
import time
from pathlib import Path
from same_device import ROOT, capture_pairs, qa_leaf_gradients, visual_backward, cache_bytes
from profile_delayed_qa import (PREP, WEIGHTS, CHECKPOINT, runtime, make_builder,
                                GroupScorer, read, sha, decode, run_case, question_text)
import torch

NEW_PREP=ROOT/'runs/delayed_recompute_v2_prepare_20261104_01'


def reset(device):
    gc.collect()
    if device.type=='cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def compare(actual,reference):
    result={}
    for name,ref in reference.items():
        val=actual[name]
        delta=(val-ref).abs()
        relative=float((val-ref).norm()/ref.norm().clamp_min(1e-12))
        bad=int((delta > 2e-6+5e-4*ref.abs()).sum())
        result[name]=dict(max_abs=float(delta.max()),relative_l2=relative,
                          mismatched_elements=bad,elements=ref.numel(),
                          passed=bad==0 and relative<=5e-4 and bool(torch.isfinite(val).all()))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-tiny',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_tiny:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
        assert torch.cuda.device_count()==1
    for path,digest in read(NEW_PREP/'execution_lock.json').items():assert sha(ROOT/path)==digest,path
    cases=read(PREP/'cases.json')
    assert len(cases)==2 and sum(len(c['audit']['samples'])//2 for c in cases)==38
    args.out.mkdir()
    env=runtime()
    processor=env['AutoProcessor'].from_pretrained(WEIGHTS,local_files_only=True)
    builder=make_builder(env,processor)
    if args.cpu_tiny:
        config=env['tiny_config'](processor);config['vision_config']['depth']=24
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config,attn_implementation='sdpa')).float()
    else:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(WEIGHTS,local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False)
    versions={n:p._version for n,p in model.named_parameters()}
    device=model.get_input_embeddings().weight.device
    scorer=GroupScorer(model.config.vision_config.hidden_size).to(device)
    if not args.cpu_tiny:scorer.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    all_passed=True
    with (args.out/'cases.jsonl').open('x',encoding='utf-8') as log:
        for case in cases:
            source=case['source'];assert source['role']=='train'
            path=ROOT/'datasets/learned-prune-nextqa-v1'/source['mirror_path']
            assert sha(path)==source['mirror_lfs_sha256']
            pairs=decode(env,path,case['audit'],builder)
            arms,grads={},{}
            for name in ('whole_first','whole_repeat'):
                scorer.zero_grad(set_to_none=True)
                for _,_,batch in pairs:batch.to('cpu')
                reset(device)
                with env['fp32_reference']():arms[name]=run_case(model,builder,scorer,pairs,case)
                grads[name]={n:p.grad.detach().cpu().clone() for n,p in scorer.named_parameters()}
            scorer.zero_grad(set_to_none=True)
            for _,_,batch in pairs:batch.to('cpu')
            inputs=[(a,b,batch.pixel_values_videos,batch.video_grid_thw) for a,b,batch in pairs]
            reset(device);started=time.monotonic()
            with env['fp32_reference']():
                captured=capture_pairs(model.model.visual,scorer,inputs)
                observed=qa_leaf_gradients(model,builder,captured,question_text(case['question']),case['question']['answer_index'])
                score_grads=visual_backward(model.model.visual,scorer,captured,observed['gradients'],inputs.__getitem__)
            if device.type=='cuda':torch.cuda.synchronize()
            arms['recompute']=dict(ce=observed['loss'],seconds=time.monotonic()-started,
                old_score_grad_l1=sum(float(score_grads[i].abs().sum()) for i in case['evidence_pairs']),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else None,
                cache_tensor_bytes=cache_bytes(captured),projected_leaf_bytes=observed['projected_leaf_bytes'],
                leaf_gradient_bytes=sum(g.numel()*g.element_size() for g in observed['gradients'].values()),
                replay_device=observed['replay_device'])
            grads['recompute']={n:p.grad.detach().cpu().clone() for n,p in scorer.named_parameters()}
            comparisons={name:compare(grads[name],grads['whole_first']) for name in ('whole_repeat','recompute')}
            passed=all(v['passed'] for row in comparisons.values() for v in row.values())
            passed=passed and all(abs(a['ce']-arms['whole_first']['ce'])<=2e-4 for a in arms.values())
            passed=passed and arms['recompute']['old_score_grad_l1']>0
            assert observed['metadata']['visual_tokens']==arms['whole_first']['visual_tokens']
            assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
            # Save ALL diagnostics before applying the numerical gate, including failures.
            torch.save(grads,args.out/(case['question']['id']+'-gradients.pt'))
            row=dict(id=case['question']['id'],arms=arms,comparisons=comparisons,passed=passed,
                     gradient_atol=2e-6,gradient_rtol=5e-4,relative_l2_limit=5e-4,updates=0)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            all_passed=all_passed and passed
            del captured,observed,score_grads,grads,inputs,pairs
    report=dict(status='passed' if all_passed else 'numerical_gate_failed',cpu_tiny=args.cpu_tiny,
        cases=2,visual_forwards=152,language_forwards=10,qa_backwards=6,recomputed_visual_backwards=38,
        updates=0,generated_tokens=0,gradient_thresholds_unchanged=True,
        limits='fixed two-case numerical diagnostic; no inference accuracy or training throughput claim')
    (args.out/'completed.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not all_passed:raise SystemExit('Numerical diagnostic finished; fixed gate failed. No updates permitted.')


if __name__=='__main__':main()
