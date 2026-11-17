"""7B utility targets and held-source evaluation; all subsets are diagnostic data.

Evaluation answers are recorded before fitting but never fed to optimizer or
epoch selection. The fitted policy selects using question/recent/bank features.
"""
import argparse,os,time,gc,traceback
from common import *
from selector import *
from run_prune import check_pair

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and not out.exists();verify();out.mkdir()
    start=time.monotonic();counts=dict(questions=0,scoring_forwards=0,representations=0,vision=0)
    def guard():
        if time.monotonic()-start>237*60:raise TimeoutError('fixed 237 minute stop')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<192*2**20
    try:
        model,p=load_model();versions={n:x._version for n,x in model.named_parameters()}
        groups=read(PREP/'fit_sources.json');tasks=read(PREP/'evaluation.json');traces=read(PREP/'traces.json')
        gs={}
        for q in tasks:gs.setdefault((q['suite'],q['source_id']),[]).append(q)
        groups += [dict(role='evaluation',source_id=source,media_path=qs[0]['media_path'],media_sha256=qs[0]['media_sha256'],questions=qs,trace_key=suite+'|'+source) for (suite,source),qs in sorted(gs.items())]
        features=[]
        with torch.no_grad(),(out/'scores.jsonl').open('x',encoding='utf-8') as log:
            for group in groups:
                guard();qs=sorted(group['questions'],key=lambda q:(q['query_time_s'],q['id']));media=path(qs[0]);assert sha(media)==group['media_sha256']
                memory=Memory();index=0
                def commit(first,second,padding=False):
                    nonlocal index
                    guard()
                    if 'trace_key' in group:check_pair(traces[group['trace_key']],index,first,second)
                    batch=readout.pair_inputs(p,[first.rgb,second.rgb],device='cuda')
                    values=model.get_video_features(**batch,return_dict=True).pooler_output[0]
                    memory.append(values,torch.arange(140),index,first.timestamp,second.timestamp)
                    index+=1;counts['vision']+=1
                driver=CausalPairs(media,commit)
                try:
                    for q in qs:
                        driver.advance(q['query_time_s']);guard();recent,history=memory.parts();before=digest(memory.bank.nodes())
                        choices,mask=choices_mask(len(history))
                        query=forward(model,p,recent,q['question'],representation=True);counts['representations']+=1
                        f=descriptor_features(query,recent,history)
                        f.update(id=q['id'],source_id=q['source_id'],role=group['role'],suite=q['suite'],question=q['question'],choices=choices,
                            history_nodes=[n.metadata() for n in history],recent_nodes=[n.metadata() for n in recent])
                        features.append(f)
                        for subset,choice in enumerate(choices):
                            nodes=[history[i] for i in choice]+recent
                            for shift in (0,2):
                                guard();logits=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                                row=dict(id=q['id'],source_id=q['source_id'],role=group['role'],suite=q['suite'],subset=subset,choice=choice,
                                    shift=shift,option_logits=logits.tolist(),selected_original=(int(logits.argmax())+shift)%len(q['options']),
                                    visual_tokens=sum(len(n.features) for n in nodes))
                                log.write(json.dumps(row,ensure_ascii=False)+'\n');counts['scoring_forwards']+=1
                        assert digest(memory.bank.nodes())==before
                        log.flush();counts['questions']+=1
                        print(json.dumps(dict(id=q['id'],role=group['role'],elapsed=time.monotonic()-start,**counts)),flush=True)
                finally:driver.close()
                # Durable feature checkpoint after every source; no implicit rerun.
                torch.save(features,out/'features.pt')
                del memory;gc.collect();torch.cuda.empty_cache()
        plan=read(PREP/'plan.json')
        assert counts['questions']==plan['history_representations'] and counts['scoring_forwards']<=plan['history_scoring_max']
        assert all(x._version==versions[n] and x.grad is None for n,x in model.named_parameters())
        del model;gc.collect();torch.cuda.empty_cache()
        save(out/'collection_completed.json',dict(**counts,seconds=time.monotonic()-start,base_updates=0,generated_tokens=0))
        from fit_selector import fit_and_evaluate
        result=fit_and_evaluate(out);verify()
        save(out/'completed.json',dict(**counts,seconds=time.monotonic()-start,base_updates=0,generated_tokens=0,trained_heads=2,epochs=20,metrics=result))
    except BaseException as e:
        save(out/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),**counts));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out)
