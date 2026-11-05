"""Three fixed four-update trajectories. Engineering check, not capability eval."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-delayed-recompute-v2'))
from same_device import capture_pairs,qa_leaf_gradients,visual_backward
from profile_delayed_qa import PREP,WEIGHTS,CHECKPOINT,runtime,make_builder,GroupScorer,read,sha,decode,run_case,question_text
from profile_recompute_device import reset
from delayed_qa import last_logits
from resolution_runtime import context
from adapter import sparse_pair
from objective import loss_parts,option_ids
import torch
NEW_PREP=ROOT/'runs/recompute_updates_prepare_20261104_01'


def snapshot(scorer):return {n:p.detach().cpu().clone() for n,p in scorer.state_dict().items()}


def inputs_for(pairs):
    for _,_,batch in pairs:batch.to('cpu')
    return [(a,b,batch.pixel_values_videos,batch.video_grid_thw) for a,b,batch in pairs]


@torch.no_grad()
def probe(model,builder,scorer,pairs,case):
    captured=capture_pairs(model.model.visual,scorer,inputs_for(pairs))
    recent=[sparse_pair(p.features,p.ids,i,pairs[i][0],pairs[i][1])
            for i,p in captured.pairs.items() if i>=len(pairs)-8]
    results={}
    for name,nodes in [('LS',captured.nodes),('recent',recent)]:
        emb,pos,ids,meta=context(model,builder,nodes,question_text(case['question']))
        logits=last_logits(model,emb,pos)
        options=option_ids(builder,logits.device)
        loss,*_=loss_parts(logits,options,case['question']['answer_index'])
        values=logits[0,options].detach().cpu()
        results[name]=dict(ce=float(loss),option_logits=values.tolist(),
            option_argmax=int(values.argmax()),visual_tokens=meta['visual_tokens'])
    return dict(id=case['question']['id'],readouts=results,
                selected_ids={str(i):p.ids.tolist() for i,p in captured.pairs.items()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-tiny',action='store_true')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    torch.set_num_threads(1);torch.manual_seed(20261104)
    if args.cpu_tiny:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
        assert torch.cuda.device_count()==1
    for name,digest in read(NEW_PREP/'execution_lock.json').items():assert sha(ROOT/name)==digest,name
    cases=read(PREP/'cases.json');assert len(cases)==2
    env=runtime();processor=env['AutoProcessor'].from_pretrained(WEIGHTS,local_files_only=True)
    builder=make_builder(env,processor)
    if args.cpu_tiny:
        config=env['tiny_config'](processor);config['vision_config']['depth']=24
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config,attn_implementation='sdpa')).float()
    else:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(WEIGHTS,local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
    device=model.get_input_embeddings().weight.device
    scorer=GroupScorer(model.config.vision_config.hidden_size).to(device)
    if not args.cpu_tiny:scorer.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    initial=snapshot(scorer);all_pairs=[]
    for case in cases:
        source=case['source'];assert source['role']=='train'
        path=ROOT/'datasets/learned-prune-nextqa-v1'/source['mirror_path'];assert sha(path)==source['mirror_lfs_sha256']
        all_pairs.append(decode(env,path,case['audit'],builder))
    assert sum(map(len,all_pairs))==38
    args.out.mkdir();probes={};endpoints={};trajectories={}
    with env['fp32_reference']():
        probes['initial']=[probe(model,builder,scorer,p,c) for p,c in zip(all_pairs,cases)]
        for arm in ('whole_first','whole_repeat','recompute'):
            scorer.load_state_dict(initial)
            opt=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
            rows=[]
            with (args.out/(arm+'_updates.jsonl')).open('x',encoding='utf-8') as log:
                for step,index in enumerate((0,1,0,1),1):
                    pairs=all_pairs[index];case=cases[index]
                    opt.zero_grad(set_to_none=True);inputs=inputs_for(pairs);reset(device);started=time.monotonic()
                    if arm=='recompute':
                        captured=capture_pairs(model.model.visual,scorer,inputs)
                        qa=qa_leaf_gradients(model,builder,captured,question_text(case['question']),case['question']['answer_index'])
                        visual_backward(model.model.visual,scorer,captured,qa['gradients'],inputs.__getitem__)
                        ce=qa['loss'];del captured,qa
                    else:ce=run_case(model,builder,scorer,pairs,case)['ce']
                    norm=torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.)
                    assert torch.isfinite(norm) and float(norm)>0
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scorer.parameters())
                    assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
                    opt.step()
                    if device.type=='cuda':torch.cuda.synchronize()
                    state=snapshot(scorer);assert all(torch.isfinite(p).all() for p in state.values())
                    torch.save(state,args.out/f'{arm}_step{step}.pt')
                    row=dict(step=step,id=case['question']['id'],ce=ce,gradient_norm=float(norm),
                        seconds=time.monotonic()-started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
                    log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
            trajectories[arm]=rows;endpoints[arm]=snapshot(scorer)
            assert any(not torch.equal(initial[n],endpoints[arm][n]) for n in initial)
            scorer.zero_grad(set_to_none=True);del opt
            probes[arm]=[probe(model,builder,scorer,p,c) for p,c in zip(all_pairs,cases)]
    comparisons={}
    for arm in ('whole_repeat','recompute'):
        delta=torch.cat([(endpoints[arm][n]-endpoints['whole_first'][n]).flatten().double() for n in sorted(initial)])
        update=torch.cat([(endpoints['whole_first'][n]-initial[n]).flatten().double() for n in sorted(initial)])
        ce_deltas=[abs(probes[arm][i]['readouts'][r]['ce']-probes['whole_first'][i]['readouts'][r]['ce']) for i in range(2) for r in ('LS','recent')]
        comparisons[arm]=dict(parameter_delta_l2=float(delta.norm()),parameter_delta_max=float(delta.abs().max()),
            relative_to_update_l2=float(delta.norm()/update.norm().clamp_min(1e-12)),max_probe_ce_delta=max(ce_deltas),
            choice_changes=sum(probes[arm][i]['readouts'][r]['option_argmax']!=probes['whole_first'][i]['readouts'][r]['option_argmax'] for i in range(2) for r in ('LS','recent')))
    # New functional screening thresholds, not a replacement for the failed old gradient gate.
    decision=all(v['max_probe_ce_delta']<=.01 and v['relative_to_update_l2']<=.1 for v in comparisons.values())
    report=dict(status='completed',functional_screen_passed=decision,comparisons=comparisons,
        cpu_tiny=args.cpu_tiny,updates=12,visual_forwards=456,language_forwards=36,generated_tokens=0,
        probe_semantics='Same two training questions through LS/recent; NOT independently labeled current questions or held-out validation',
        old_gradient_gate_remains_failed=True,limits='4 updates per arm; no generalization claim')
    for name,value in [('probes.json',probes),('completed.json',report)]:
        (args.out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
