"""One fixed six-arm QA evaluation; all trained-head selection overhead is timed."""
import argparse,gc,os,time,json,hashlib
from pathlib import Path
import numpy as np
from selector_qa_common import ROOT,PREP,HEADS,ARMS,read,save,sha,shared,load_heads,descriptors,choose,media_path

def main(out):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':
        raise RuntimeError('Explicit approved Slurm allocation required')
    if out.exists() or not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('New owned output required')
    from selector_qa_freeze import verify
    verify()
    base,env,make_builder,Bank,distance,pair_node,digest,extract=shared()
    from canonical_prompt import question_prompt,rotate
    from temporal_policy import lexical_route
    torch=env['torch'];psutil=env['psutil'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<24*2**30 or psutil.virtual_memory().available<12*2**30:
        raise RuntimeError('GPU/system memory margin required')
    protocol=read(PREP/'protocol.json');plan=read(PREP/'plan.json');tasks=read(PREP/'requests.json')
    checkpoints={r['task_id']:r for r in plan['checkpoints']}
    refs={(r['task_id'],r['arm'],r['shift']):r for r in read(PREP/'references_LOCAL_ONLY.json')}
    for s in plan['sources']:
        p=media_path(s)
        if p.stat().st_size!=s['bytes'] or sha(p)!=s['sha256']:raise ValueError('Read-only media SHA mismatch')
    # Cached features only validate the same already-exported 36 queries; never supply inference values for free.
    with np.load(ROOT/'runs/selector-features-219523/features.npz',allow_pickle=False) as f:
        allids=f['ids'].tolist();ix=[i for i,x in enumerate(allids) if x in {t['id'] for t in tasks}]
        cached={allids[i]:{k:f[k][i].copy() for k in ('query','recent','history','metadata','mask')} for i in ix}
    if len(cached)!=36:raise ValueError('Expected 36 independent-head development queries')
    expected={name:{p['id']:p['history_indices'] for p in read(HEADS/(name+'_predictions.json')) if p['id'] in cached}
              for name in ('content_visible','history_blind')}
    heads=load_heads(torch)
    out.mkdir();started=time.monotonic()
    stats=dict(vision=0,formal_answers=0,warmup_answers=0,question_prefills=0,generated_tokens=0,
               reference_matches=0,long_head_decision_checks=0)
    def guard():
        if time.monotonic()-started>3450 or psutil.Process().memory_info().rss>20*2**30:
            raise RuntimeError('Wall/RSS budget exceeded')
        if torch.cuda.max_memory_allocated()>24*2**30:raise RuntimeError('Allocated GPU memory bound')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>96*2**20:raise RuntimeError('Result budget')
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError('Base weight load mismatch')
    model.eval();model.requires_grad_(False)
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    def timed_decision(nodes,t,arm):
        torch.cuda.synchronize();start=time.monotonic()
        if arm in ('content_visible','history_blind','matched_latest'):
            # Each actual answer pays its own fresh VLM question prefill plus CPU descriptors/scoring.
            q,_=extract(env,model,builder,nodes[-4:],t['question'],False);stats['question_prefills']+=1
            values=descriptors(torch,nodes,t['query_time_s']);decision=choose(torch,heads,arm,q,values)
            torch.cuda.synchronize();decision['selector_seconds']=time.monotonic()-start
            if t['id'] in cached:
                c=cached[t['id']]
                for k in ('query','recent','history','metadata'):
                    observed=q.numpy() if k=='query' else values[k][0].numpy()
                    if not np.allclose(observed,c[k],rtol=1e-5,atol=1e-5):raise ValueError('Exported selector input drift: '+k)
                if arm in expected:
                    if decision['selected_history_indices']!=expected[arm][t['id']]:raise ValueError('Frozen head decision drift')
                    stats['long_head_decision_checks']+=1
        elif arm=='lexical':
            route=lexical_route(t['question'])['route']
            decision=dict(selected_history_indices=[] if route=='recent' else [0,1,2,3],reading=route)
        else:decision=dict(selected_history_indices=[] if arm=='recent' else [0,1,2,3],reading=arm)
        if 'selector_seconds' not in decision:
            torch.cuda.synchronize();decision['selector_seconds']=time.monotonic()-start
        return decision
    delivered=set();source_costs=[];query_checks=[]
    try:
        with torch.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for source_index,s in enumerate(plan['sources']):
                scheduled=[t for t in tasks if t['source_name']==s['name']]
                bank=Bank('LS',108);previous=None;start=time.monotonic();vision_s=0.;bank_s=0.;source_QA_s=0.
                for i,(a,b) in enumerate(base.sampled_pairs(media_path(s),s,guard)):
                    guard();torch.cuda.synchronize();tick=time.monotonic();v=base.project(env,model,builder,a,b)
                    torch.cuda.synchronize();vision_s+=time.monotonic()-tick;stats['vision']+=1
                    tick=time.monotonic();desc=v.mean(0)
                    bank.append(pair_node(v,i,a.timestamp,b.timestamp,(9,12)),
                        starts_event=previous is None or distance(previous,desc)>.047536224126815796)
                    previous=desc.clone();bank_s+=time.monotonic()-tick;del v
                    for t in [t for t in scheduled if checkpoints[t['id']]['committed_pairs']==i+1]:
                        nodes=bank.nodes();before=digest(nodes);p=checkpoints[t['id']]
                        if len(nodes)!=8 or b.timestamp!=p['cutoff_s'] or b.timestamp>t['query_time_s']:raise ValueError('Causal checkpoint drift')
                        for actual,expected_node in zip(nodes,p['nodes']):
                            m=actual.metadata()
                            if any(m[k]!=expected_node[k] for k in ('first','last','begin','end','center','tokens','exact_pair')):
                                raise ValueError('Bank partition differs from completed feature/reference stage')
                        if stats['warmup_answers']==0:
                            for arm in ARMS:
                                d=timed_decision(nodes,t,arm)
                                result=base.infer(env,model,builder,nodes,question_prompt(rotate(t,0)[0]),
                                    d['selected_history_indices'],full=len(d['selected_history_indices'])==4)
                                stats['warmup_answers']+=1;stats['generated_tokens']+=len(result['generated_ids'])
                        for shift in range(5):
                            text=question_prompt(rotate(t,shift)[0])
                            # Counterbalance all methods. A reference mismatch invalidates the incomplete run.
                            rotation=(source_index+shift)%len(ARMS)
                            order=ARMS[rotation:]+ARMS[:rotation]
                            decisions={}
                            for arm in order:
                                guard();d=timed_decision(nodes,t,arm);decisions[arm]=d['selected_history_indices']
                                result=base.infer(env,model,builder,nodes,text,d['selected_history_indices'],full=len(d['selected_history_indices'])==4)
                                stats['formal_answers']+=1;stats['generated_tokens']+=len(result['generated_ids'])
                                ref=refs.get((t['id'],arm,shift))
                                if ref:
                                    if any(result[k]!=ref[k] for k in ('input_ids_sha256','position_ids_sha256','generated_ids')):
                                        raise RuntimeError('Completed reference changed; stop without resubmission')
                                    stats['reference_matches']+=1
                                total=result['QA_seconds']+d['selector_seconds'];source_QA_s+=total
                                row=dict(task_id=t['id'],evaluation_group=t['evaluation_group'],arm=arm,shift=shift,
                                    selector=d,QA_with_selection_seconds=total,**result)
                                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                            if len(decisions['matched_latest'])!=len(decisions['content_visible']):raise RuntimeError('Matched read-count drift')
                            if digest(bank.nodes())!=before:raise RuntimeError('QA/selector mutated bank')
                        delivered.add(t['id']);query_checks.append(dict(task_id=t['id'],nodes=[n.metadata() for n in nodes]))
                wall=time.monotonic()-start
                source_costs.append(dict(source=s['name'],wall_seconds_including_all_methods=wall,shared_vision_seconds=vision_s,
                    shared_bank_seconds=bank_s,measured_formal_QA_seconds_all_methods=source_QA_s,
                    unallocated_wall_includes_decode_validation_and_warmups=True))
                del bank,previous;gc.collect();torch.cuda.empty_cache()
                print(json.dumps(dict(source=s['name'],**stats)),flush=True)
        if (len(delivered)!=100 or stats['formal_answers']!=3000 or stats['warmup_answers']!=6
            or stats['vision']!=protocol['visual_forwards'] or stats['question_prefills']!=1503
            or stats['reference_matches']!=640 or stats['long_head_decision_checks']!=360
            or stats['generated_tokens']>protocol['maximum_generation_tokens']):raise RuntimeError('Frozen workload mismatch')
        verify();save(out/'completed.json',dict(status='complete',**stats,wall_seconds=time.monotonic()-started,
            source_costs=source_costs,checkpoints=query_checks,pretrained_loads=1,base_updates=0,selector_updates=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),shared_video_cost_not_standalone=True,
            automatic_retry=False))
    except Exception as e:
        save(out/'failed.json',dict(type=type(e).__name__,error=str(e),**stats));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
