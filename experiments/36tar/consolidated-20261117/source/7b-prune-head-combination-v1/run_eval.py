import argparse,os,time,gc,traceback
from combination import *

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and not out.exists();verify_all();out.mkdir()
    plan=read(HERE/'plan.json');refs={tuple(r[k] for k in ('id','mode','route','shift')):r for r in read(HERE/'references.json')}
    selections=read(HERE/'full_selections.json');traces=read(HERE/'traces.json')
    start=time.monotonic();counts=dict(questions=0,classifications=0,representations=0,vision=0,references=0,state_checks=0)
    def guard():
        if time.monotonic()-start>237*60:raise TimeoutError('237 minute internal stop; partial answers retained')
        assert counts['vision']<=plan['vision_bound']
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<128*2**20
    try:
        model,p=load_model();head=load_head();versions={n:x._version for n,x in model.named_parameters()}
        groups={}
        for q in read(HERE/'questions.json'):groups.setdefault((q['suite'],q['source_id']),[]).append(q)
        # Short sources first, so all six read routes and references execute early.
        ordered=sorted(groups.items(),key=lambda item:(min(q['query_time_s'] for q in item[1]),item[0]))
        with torch.no_grad(),(out/'answers.jsonl').open('x',encoding='utf8') as af,(out/'routes.jsonl').open('x',encoding='utf8') as rf:
            for key,qs in ordered:
                qs.sort(key=lambda q:(q['query_time_s'],q['id']));trace=traces['|'.join(key)]
                media=path(qs[0]);assert sha(media)==qs[0]['media_sha256']
                memories={m:Memory(m,archive=False) for m in plan['modes']};index=0
                def commit(a,b,padding=False):
                    nonlocal index
                    guard();assert a.rgb.shape==b.rgb.shape==NEW_CANVAS;check_pair(trace,index,a,b)
                    batch=readout.pair_inputs(p,[a.rgb,b.rgb],device='cuda')
                    full,ids,meta=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'full',capture_prefix=True);counts['vision']+=1
                    if counts['vision']==1:
                        expected=model.get_video_features(**batch,return_dict=True).pooler_output[0];counts['vision']+=1
                        torch.testing.assert_close(full,expected,rtol=0,atol=0)
                    memories['full'].append(full,ids,index,a.timestamp,b.timestamp)
                    content,cids,_=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'content',memories['content'].scorer,prefix=meta['prefix']);counts['vision']+=1
                    memories['content'].append(content,cids,index,a.timestamp,b.timestamp);index+=1
                driver=CausalPairs(media,commit)
                try:
                    for q in qs:
                        state=driver.advance(q['query_time_s']);guard()
                        if q['id'] in trace.get('states',{}):compare_state(state,trace['states'][q['id']]);counts['state_checks']+=1
                        for mode,memory in memories.items():
                            assert memory.archive is None
                            recent,history=memory.parts();before=digest(memory.bank.nodes())
                            torch.cuda.synchronize();tick=time.monotonic()
                            query=forward(model,p,recent,q['question'],representation=True);counts['representations']+=1
                            selected,choice,details=select_nodes(head,query,recent,history)
                            selection_seconds=time.monotonic()-tick
                            if mode=='full':assert choice==selections[q['id']],('frozen full-head route changed',q['id'])
                            routes={'recent8':recent,'LS':history+recent,'head':selected}
                            if mode=='content':routes['same_count_earliest']=history[:len(choice)]+recent
                            rf.write(json.dumps(dict(id=q['id'],suite=q['suite'],mode=mode,choice=choice,
                                history_available=len(history),selection_seconds=selection_seconds,**details))+'\n');rf.flush()
                            for route,nodes in routes.items():
                                for shift in (0,2):
                                    guard();torch.cuda.synchronize();tick=time.monotonic()
                                    values=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                                    row=dict(id=q['id'],source_id=q['source_id'],suite=q['suite'],mode=mode,route=route,shift=shift,
                                        option_logits=values.tolist(),selected_original=(int(values.argmax())+shift)%len(q['options']),
                                        visual_tokens=sum(len(n.features) for n in nodes),read_seconds=time.monotonic()-tick,
                                        selection_seconds=selection_seconds if route in ('head','same_count_earliest') else 0.)
                                    k=(q['id'],mode,route,shift)
                                    if k in refs:check_reference(row,refs[k]);counts['references']+=1
                                    af.write(json.dumps(row)+'\n');af.flush();counts['classifications']+=1
                            assert digest(memory.bank.nodes())==before
                        counts['questions']+=1;print(json.dumps(dict(id=q['id'],elapsed=time.monotonic()-start,**counts)),flush=True)
                finally:driver.close()
                del memories;gc.collect();torch.cuda.empty_cache()
        assert counts['questions']==210 and counts['classifications']==2940 and counts['references']==1860 and counts['representations']==420
        assert all(x._version==versions[n] and x.grad is None for n,x in model.named_parameters())
        verify_all();save(out/'completed.json',dict(**counts,seconds=time.monotonic()-start,base_updates=0,head_updates=0,generated_tokens=0,standalone_cost=False))
    except BaseException as e:
        save(out/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),**counts));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
