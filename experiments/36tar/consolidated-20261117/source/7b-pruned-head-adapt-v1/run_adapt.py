import argparse,os,time,gc,traceback
from adapt import *

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and not out.exists()
    verify_adapt();out.mkdir();plan=read(HERE/'plan.json')
    start=time.monotonic();counts=dict(train_questions=0,eval_questions=0,classifications=0,
        representations=0,vision=0,references=0,state_checks=0,route_checks=0)
    def guard():
        if time.monotonic()-start>177*60:raise TimeoutError('177 minute internal stop; no automatic resume')
        assert counts['vision']<=plan['vision_bound'] and counts['classifications']<=7732
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<192*2**20
    def progress(q,stage):
        print(json.dumps(dict(id=q['id'],stage=stage,elapsed=time.monotonic()-start,**counts)),flush=True)
    try:
        model,p=load_model();oldhead=load_head();versions={n:x._version for n,x in model.named_parameters()}
        groups=read(HERE/'fit_sources.json');features=[];scores=[]
        # Decode and encode only causal pairs. No full visual feature archive.
        def source_loop(group,visit,trace=None):
            qs=sorted(group['questions'],key=lambda q:(q['query_time_s'],q['id']))
            media=path(qs[0]);assert sha(media)==group['media_sha256']
            memory=Memory('content',archive=False);index=0
            def commit(a,b,padding=False):
                nonlocal index
                guard();assert a.rgb.shape==b.rgb.shape==NEW_CANVAS
                if trace is not None:check_pair(trace,index,a,b)
                batch=readout.pair_inputs(p,[a.rgb,b.rgb],device='cuda')
                values,ids,_=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'content',memory.scorer)
                memory.append(values,ids,index,a.timestamp,b.timestamp);index+=1;counts['vision']+=1
            driver=CausalPairs(media,commit)
            try:
                with torch.no_grad():
                    for q in qs:
                        state=driver.advance(q['query_time_s']);guard()
                        if trace is not None:
                            compare_state(state,trace['states'][q['id']]);counts['state_checks']+=1
                        recent,history=memory.parts();before=digest(memory.bank.nodes())
                        assert memory.archive is None
                        visit(q,recent,history)
                        assert digest(memory.bank.nodes())==before
            finally:driver.close()
            del memory;gc.collect();torch.cuda.empty_cache()
        with (out/'scores.jsonl').open('x',encoding='utf8') as sf:
            for group in groups:
                def collect(q,recent,history):
                    choices,_=choices_mask(len(history))
                    query=forward(model,p,recent,q['question'],representation=True);counts['representations']+=1
                    f=descriptor_features(query,recent,history)
                    f.update(id=q['id'],source_id=q['source_id'],role=group['role'],choices=choices)
                    features.append(f)
                    for subset,choice in enumerate(choices):
                        nodes=[history[i] for i in choice]+recent
                        for shift in (0,2):
                            guard();values=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                            row=dict(id=q['id'],subset=subset,shift=shift,option_logits=values.tolist(),
                                selected_original=(int(values.argmax())+shift)%len(q['options']),visual_tokens=sum(len(n.features) for n in nodes))
                            scores.append(row);sf.write(json.dumps(row)+'\n');counts['classifications']+=1
                    sf.flush();counts['train_questions']+=1;progress(q,group['role'])
                source_loop(group,collect)
                torch.save(features,out/'features.pt')
        assert counts['train_questions']==176
        save(out/'collection_completed.json',dict(**counts,seconds=time.monotonic()-start))
        guard();head=train_head(features,scores,groups,out);frozen=sha(out/'head-visible.pt')
        print(json.dumps(dict(stage='head_frozen',sha=frozen,elapsed=time.monotonic()-start)),flush=True)
        # Evaluation manifest has no answer labels. It is first opened after freezing.
        tasks=read(HERE/'questions.json');traces=read(HERE/'traces.json')
        refs={(r['id'],r['route'],r['shift']):r for r in read(HERE/'references.json')}
        oldchoices=read(HERE/'old_choices.json');by_source={}
        for q in tasks:by_source.setdefault((q['suite'],q['source_id']),[]).append(q)
        with (out/'answers.jsonl').open('x',encoding='utf8') as af,(out/'routes.jsonl').open('x',encoding='utf8') as rf:
            for key,qs in sorted(by_source.items(),key=lambda item:(min(q['query_time_s'] for q in item[1]),item[0])):
                group=dict(questions=qs,media_sha256=qs[0]['media_sha256'])
                def evaluate(q,recent,history):
                    torch.cuda.synchronize();tick=time.monotonic()
                    query=forward(model,p,recent,q['question'],representation=True);counts['representations']+=1
                    query_seconds=time.monotonic()-tick
                    selected,choice,detail=select_nodes(head,query,recent,history)
                    oldnodes,oldchoice,_=select_nodes(oldhead,query,recent,history)
                    assert oldchoice==oldchoices[q['id']],('old route changed',q['id']);counts['route_checks']+=1
                    rf.write(json.dumps(dict(id=q['id'],suite=q['suite'],choice=choice,old_choice=oldchoice,
                        query_seconds=query_seconds,**detail))+'\n');rf.flush()
                    routes={'recent8':recent,'LS':history+recent,'head':oldnodes,
                        'adapted':selected,'adapted_same_count_earliest':history[:len(choice)]+recent}
                    for route,nodes in routes.items():
                        for shift in (0,2):
                            guard();torch.cuda.synchronize();tick=time.monotonic()
                            values=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                            row=dict(id=q['id'],source_id=q['source_id'],suite=q['suite'],route=route,shift=shift,
                                option_logits=values.tolist(),selected_original=(int(values.argmax())+shift)%len(q['options']),
                                visual_tokens=sum(len(n.features) for n in nodes),read_seconds=time.monotonic()-tick)
                            k=q['id'],route,shift
                            if k in refs:check_reference(row,refs[k]);counts['references']+=1
                            af.write(json.dumps(row)+'\n');af.flush();counts['classifications']+=1
                    counts['eval_questions']+=1;progress(q,'evaluation')
                source_loop(group,evaluate,traces['|'.join(key)])
        assert counts['eval_questions']==210 and counts['references']==1260 and counts['route_checks']==210
        assert counts['representations']==386 and counts['state_checks']==210
        assert sha(out/'head-visible.pt')==frozen
        assert all(x._version==versions[n] and x.grad is None for n,x in model.named_parameters())
        verify_adapt();save(out/'completed.json',dict(**counts,seconds=time.monotonic()-start,
            base_updates=0,head_updates=180,generated_tokens=0,head_sha=frozen))
    except BaseException as e:
        save(out/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),**counts));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
