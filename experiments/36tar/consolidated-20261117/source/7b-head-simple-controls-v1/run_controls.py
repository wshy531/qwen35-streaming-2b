import sys,os,time,gc,traceback,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-head-stability-v1'))
from run_eval import *
PREP=ROOT/'runs/7b_head_simple_prepare_20261112_01'
def verify_controls():
    check_new()
    for rel,h in read(PREP/'lock.json').items():assert sha(ROOT/rel)==h,rel
def main(out):
    assert os.environ.get('SLURM_JOB_ID') and not out.exists()
    verify_controls();out.mkdir();plan=read(PREP/'plan.json');tasks=read(PREP/'tasks.json')
    counts=dict(questions=0,classifications=0,vision=0,references=0,states=0,reused=0)
    start=time.monotonic()
    def guard():
        if time.monotonic()-start>177*60:raise TimeoutError('internal stop')
        assert counts['vision']<=plan['vision_bound'] and counts['classifications']<=plan['max_scores']
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<96*2**20
    try:
        model,p=load_model();versions={n:x._version for n,x in model.named_parameters()}
        traces=read(HERE/'traces.json');groups={}
        for t in tasks:
            q=t['question'];groups.setdefault((q['suite'],q['source_id']),[]).append(t)
        with (out/'answers.jsonl').open('x',encoding='utf8') as af,(out/'checks.jsonl').open('x',encoding='utf8') as cf:
            for key,ts in sorted(groups.items(),key=lambda item:(min(t['question']['query_time_s'] for t in item[1]),item[0])):
                q=ts[0]['question'];media=path(q);assert sha(media)==q['media_sha256']
                memory=Memory('content',archive=False);index=0;trace=traces['|'.join(key)]
                def commit(a,b,padding=False):
                    nonlocal index
                    guard();assert a.rgb.shape==b.rgb.shape==NEW_CANVAS;check_pair(trace,index,a,b)
                    batch=readout.pair_inputs(p,[a.rgb,b.rgb],device='cuda')
                    values,ids,_=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'content',memory.scorer)
                    memory.append(values,ids,index,a.timestamp,b.timestamp);index+=1;counts['vision']+=1
                driver=CausalPairs(media,commit)
                try:
                    with torch.no_grad():
                        for t in sorted(ts,key=lambda t:(t['question']['query_time_s'],t['question']['id'])):
                            q=t['question'];state=driver.advance(q['query_time_s']);guard()
                            compare_state(state,trace['states'][q['id']]);counts['states']+=1
                            recent,history=memory.parts();assert len(history)==t['n_history'] and memory.archive is None
                            before=digest(memory.bank.nodes())
                            def score(choice,shift):
                                guard();nodes=[history[i] for i in choice]+recent
                                torch.cuda.synchronize();tick=time.monotonic()
                                v=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                                counts['classifications']+=1
                                return dict(id=q['id'],suite=q['suite'],source_id=q['source_id'],choice=choice,shift=shift,
                                    option_logits=v.tolist(),selected_original=(int(v.argmax())+shift)%len(q['options']),
                                    visual_tokens=sum(len(n.features) for n in nodes),read_seconds=time.monotonic()-tick)
                            if t['reference']:
                                ref=t['reference'];r=score(ref['choice'],ref['row']['shift']);check_reference(r,ref['row'])
                                cf.write(json.dumps(r)+'\n');cf.flush();counts['references']+=1
                            scored={}
                            for e in t['cached']:
                                row=dict(e['row']);row.update(choice=e['choice'],provenance='reused',source=e['source'],source_sha=e['sha'])
                                nodes=[history[i] for i in e['choice']]+recent
                                assert row['visual_tokens']==sum(len(n.features) for n in nodes)
                                scored[tuple(e['choice']),row['shift']]=row;counts['reused']+=1
                            for c in t['todo']:
                                for shift in (0,2):
                                    r=score(c,shift);r['provenance']='new';scored[tuple(c),shift]=r
                            for method,c in t['methods'].items():
                                for shift in (0,2):
                                    row=dict(scored[tuple(c),shift]);row['route']=method
                                    af.write(json.dumps(row)+'\n')
                            af.flush();assert digest(memory.bank.nodes())==before
                            counts['questions']+=1;print(json.dumps(dict(elapsed=time.monotonic()-start,**counts)),flush=True)
                finally:driver.close()
                del memory;gc.collect();torch.cuda.empty_cache()
        assert counts['questions']==counts['states']==210
        assert counts['classifications']==plan['max_scores'] and counts['references']==plan['reference_scores']
        assert counts['reused']==plan['reused_unique_scores']
        assert all(x._version==versions[n] and x.grad is None for n,x in model.named_parameters())
        verify_controls();save(out/'completed.json',dict(**counts,seconds=time.monotonic()-start,generated_tokens=0,training_updates=0))
    except BaseException as e:
        save(out/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),**counts));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
