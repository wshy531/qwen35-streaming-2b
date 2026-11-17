"""Frozen 78-question transfer, plus native-vs-streamed full-prefix audit."""
import argparse,os,time,gc,traceback
from common import *
from selector import forward,prompt,option_ids

@torch.no_grad()
def audit(model,processor,memory,q,frames,out,limit=128):
    nodes=memory.select('complete');prompt=readout.question(q,0,'B')
    ref=read(PREP/'audit_references.json')[q['id']];mode=ref['mode']
    emb,pos,ids=segmented.context(model,processor,nodes,prompt,mode)
    assert ids.numel()<=16384
    import numpy as np
    # Official multi-video interface: each already-received real pair is a clip.
    text=processor.tokenizer.decode(ids[0],skip_special_tokens=False)
    videos=[np.stack(frames[i:i+2]) for i in range(0,len(frames),2)]
    markers=segmented.vision_markers(processor)
    original=readout.template(processor,prompt)
    marker=markers[0]+processor.video_token+markers[1]
    blocks=''.join((segmented.label(n) if mode=='timestamped' else '')+marker for n in nodes)
    text=original.replace(marker,blocks)
    native=processor(text=[text],videos=videos,
        video_metadata=[dict(total_num_frames=2,fps=1.,frames_indices=[0,1]) for _ in videos],
        do_resize=False,do_sample_frames=False,return_tensors='pt').to(emb.device)
    expected,_=model.model.get_rope_index(native['input_ids'],video_grid_thw=native['video_grid_thw'],second_per_grid_ts=native['second_per_grid_ts'])
    assert torch.equal(ids,native['input_ids']) and torch.equal(pos,expected)
    # Independent official whole-video visual forward. It uses the SAME pixels,
    # temporal grid, prompt and coordinates as the incremental path.
    visual=torch.cat(model.get_video_features(pixel_values_videos=native['pixel_values_videos'],video_grid_thw=native['video_grid_thw'],return_dict=True).pooler_output,0)
    pair=torch.cat([n.features for n in nodes]).to(emb.device)
    visual_diff=float((visual-pair).abs().max())
    visual_close=bool(torch.isclose(visual,pair,atol=1e-4,rtol=1e-4).all())
    # Same projected features isolate language chunking from visual differences.
    whole=model.lm_head(model.model.language_model(inputs_embeds=emb,position_ids=pos,attention_mask=torch.ones(ids.shape,device=emb.device,dtype=torch.long),use_cache=False).last_hidden_state[:,-1:])[0,-1].float().cpu()
    own,stream=readout.generate_context(model,emb,pos,limit=limit)
    del visual,pair
    model.model.rope_deltas=None
    generated=model.generate(**native,max_new_tokens=limit,do_sample=False)
    native_ids=generated[0,ids.shape[1]:].tolist()
    # Separate protocol control: same frames and question as one normal video.
    whole_video=readout.native_inputs(processor,frames,prompt).to(emb.device)
    model.model.rope_deltas=None
    single_generation=model.generate(**whole_video,max_new_tokens=limit,do_sample=False)
    single_ids=single_generation[0,whole_video['input_ids'].shape[1]:].tolist()
    diff=float((whole-stream).abs().max());tv=float((whole.softmax(-1)-stream.softmax(-1)).abs().sum()/2)
    torch.save(dict(whole=whole,chunk512=stream),out/f"audit-logits-{hashlib.sha256(q['id'].encode()).hexdigest()[:12]}.pt")
    return dict(id=q['id'],suite=q['suite'],mode=mode,input_tokens=ids.numel(),input_sha=tsha(ids),position_sha=tsha(pos),
        old_generated_ids_exact=own==ref['generated_ids'],
        native_input_and_positions_exact=True,visual_max_abs=visual_diff,visual_close=visual_close,
        language_max_abs=diff,total_variation=tv,native_ids=native_ids,stream_ids=own,
        single_video_native_ids=single_ids,single_video_native_tokens=int(whole_video['input_ids'].numel()),
        native_text=processor.tokenizer.decode(native_ids,skip_special_tokens=True),
        stream_text=processor.tokenizer.decode(own,skip_special_tokens=True),
        single_video_native_text=processor.tokenizer.decode(single_ids,skip_special_tokens=True),
        passed=visual_close and diff<=1e-3 and tv<=1e-4 and native_ids==own)

def check_pair(spec,i,first,second):
    if isinstance(spec.get('pairs'),list):
        r=spec['pairs'][i]
        for p,k in [(first,'first'),(second,'second')]:
            assert p.frame_id==r[k]['frame_id'] and abs(p.timestamp-r[k]['pts_s'])<1e-6
            if 'rgb_sha256' in r[k]:assert hashlib.sha256(p.rgb.tobytes()).hexdigest()==r[k]['rgb_sha256']
    elif spec.get('sampled_trace'):
        for p,r in zip((first,second),spec['sampled_trace'][2*i:2*i+2]):
            assert p.frame_id==r['frame_id'] and abs(p.timestamp-r['pts_s'])<1e-6

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and not out.exists();verify();out.mkdir()
    start=time.monotonic();counts=dict(answers=0,vision=0,audits=0,regressions=0,generated_tokens=0,pairs=0)
    plan=read(PREP/'plan.json');audit_ids=set(read(PREP/'audit_ids.json'));refs=read(PREP/'references.json')
    def guard():
        if time.monotonic()-start>357*60:raise TimeoutError('fixed 357 minute internal stop')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<128*2**20
    try:
        model,p=load_model();versions={n:x._version for n,x in model.named_parameters()}
        qs=read(PREP/'evaluation.json');traces=read(PREP/'traces.json');groups={}
        for q in qs:groups.setdefault((q['suite'],q['source_id']),[]).append(q)
        with torch.no_grad(),(out/'answers.jsonl').open('x',encoding='utf-8') as log,(out/'audits.jsonl').open('x',encoding='utf-8') as af:
            for key,tasks in sorted(groups.items()):
                guard();tasks.sort(key=lambda q:(q['query_time_s'],q['id']));media=path(tasks[0]);assert sha(media)==tasks[0]['media_sha256']
                spec=traces['|'.join(key)];has_audit=any(q['id'] in audit_ids for q in tasks)
                memories={m:Memory(m,archive=has_audit and m=='full') for m in plan['prune_modes']}
                frames=[];index=0
                def commit(first,second,padding=False):
                    nonlocal index
                    guard();check_pair(spec,index,first,second)
                    batch=readout.pair_inputs(p,[first.rgb,second.rgb],device='cuda')
                    official,full_ids,full_meta=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'full',capture_prefix=True);counts['vision']+=1
                    if counts['pairs']==0:
                        keep=model.get_video_features(**batch,return_dict=True).pooler_output[0]
                        torch.testing.assert_close(keep,official,atol=0,rtol=0)
                        save(out/'all_keep_gate.json',dict(passed=True,max_abs=0));counts['vision']+=1
                    for mode,m in memories.items():
                        if mode=='full':value,ids=official,torch.arange(140,device='cuda')
                        else:
                            value,ids,_=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],mode,m.scorer,prefix=full_meta['prefix']);counts['vision']+=1
                        m.append(value,ids,index,first.timestamp,second.timestamp)
                    if has_audit:frames.extend([first.rgb.copy(),second.rgb.copy()])
                    index+=1;counts['pairs']+=1
                driver=CausalPairs(media,commit)
                try:
                    for q in tasks:
                        state=driver.advance(q['query_time_s']);guard()
                        if spec.get('states',{}).get(q['id']):assert state==spec['states'][q['id']]
                        for mode,m in memories.items():
                            for route in plan['routes']:
                                guard();torch.cuda.synchronize();tick=time.monotonic()
                                nodes=m.select(route);before=digest(nodes)
                                if mode=='full' and route=='LS' and q['id'] in audit_ids:
                                    result=answer(model,p,nodes,q);ref=refs[q['id']+'|LS']
                                    for k in ('generated_ids','visual_tokens','input_tokens','nodes'):assert result[k]==ref[k],(q['id'],route,k)
                                    counts['generated_tokens']+=len(result['generated_ids'])
                                    counts['regressions']+=1
                                for shift in (0,2):
                                    values=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                                    result=dict(id=q['id'],suite=q['suite'],source_id=q['source_id'],mode=mode,route=route,shift=shift,
                                        option_logits=values.tolist(),selected_original=(int(values.argmax())+shift)%len(q['options']),
                                        visual_tokens=sum(len(n.features) for n in nodes),nodes=[n.metadata() for n in nodes])
                                    log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush();counts['answers']+=1
                                assert digest(nodes)==before
                        if q['id'] in audit_ids:
                            r=audit(model,p,memories['full'],q,frames,out);af.write(json.dumps(r)+'\n');af.flush()
                            counts['audits']+=1;counts['vision']+=3;counts['generated_tokens']+=len(r['native_ids'])+len(r['stream_ids'])+len(r['single_video_native_ids'])
                            # A mismatch is evidence to inspect, not a reason to hide
                            # all other independent audits. No tolerance is changed.
                        print(json.dumps(dict(id=q['id'],elapsed=time.monotonic()-start,**counts)),flush=True)
                finally:driver.close()
                del memories,frames;gc.collect();torch.cuda.empty_cache()
        assert counts['answers']==plan['prune_classifications'] and counts['audits']==12 and counts['regressions']==12
        assert counts['vision']<=plan['visual_calls_prune']+1 and counts['generated_tokens']<=plan['generated_tokens_max']
        assert all(x._version==versions[n] and x.grad is None for n,x in model.named_parameters())
        verify();save(out/'completed.json',dict(**counts,seconds=time.monotonic()-start,base_updates=0,quality_protocol='two option permutations; normalized single-letter scoring',shared_first16_visual_layers=True,standalone_cost=False))
    except BaseException as e:
        save(out/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),**counts));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out)
