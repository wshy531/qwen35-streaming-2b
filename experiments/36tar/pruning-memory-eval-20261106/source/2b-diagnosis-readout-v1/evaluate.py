"""One fixed evidence-readout diagnostic. No labels, updates or full feature archive."""
import argparse,gc,os,time
from diagnosis_runtime import *

def verify_lock():
    for name,value in read(PREP/'execution_lock.json').items():
        path=ROOT/name
        if name.startswith('repo/') and not path.exists():path=ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(path)==value,name

def main(args):
    verify_lock()
    auth=read(PREP/'authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    assert auth['GPUs']==1 and auth['minutes']==20
    left,right=read(PREP/'CPU_local.json'),read(PREP/'CPU_cluster.json')
    for key in ('passed','layouts','layout_sha256','maximum_context','tiny_answers','media','selected_pairs','pixel_sha256'):
        assert left[key]==right[key],key
    assert left['passed'] and left['media']==11 and left['selected_pairs']==159
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve()) and not args.out.exists()
    args.out.mkdir();start=time.monotonic()
    counts=dict(answers=0,warmup_answers=0,vision=0,warmup_vision=0,tokens=0,sources=0)
    plan=read(PREP/'plan.json');questions=read(PREP/'questions.json');sources=read(PREP/'sources.json')
    def guard():
        if time.monotonic()-start>18*60:raise TimeoutError('fixed deadline; no automatic retry')
        assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())<32*2**20
    try:
        torch.set_num_threads(1)
        env=runtime();assert torch.cuda.device_count()==1
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        selector=CoverageScorer()
        with torch.no_grad(),env['fp32_reference'](),(args.out/'answers.jsonl').open('x',encoding='utf-8') as output:
            first_source=sources[0];it=selected_pairs(env,ROOT/first_source['media_path'],first_source)
            try:index,a,b=next(it)
            finally:it.close()
            batch,_=builder.pair(a,b,0,0);batch=batch.to('cuda')
            features,ids,_=encode_pair(model,batch,'full',None,None,selector)
            node=sparse_pair(features,ids,index,a.timestamp,b.timestamp);counts['warmup_vision']=1
            for arm in ARMS:
                guard();q=next(q for q in questions if q['source_id']==first_source['source_id'])
                result=answer(env,model,builder,[] if arm=='text' else [node],question(q,0,'B'))
                counts['warmup_answers']+=1;counts['tokens']+=len(result['generated_ids'])
            del node,batch,features,ids
            for source in sources:
                guard();path=ROOT/source['media_path'];assert sha(path)==source['media_sha256']
                nodes={}
                for index,a,b in selected_pairs(env,path,source):
                    guard();batch,_=builder.pair(a,b,0,0);batch=batch.to('cuda')
                    features,ids,_=encode_pair(model,batch,'full',None,None,selector)
                    assert len(features)==108 and ids.tolist()==list(range(108))
                    nodes[index]=sparse_pair(features,ids,index,a.timestamp,b.timestamp)
                    counts['vision']+=1
                    del batch,features,ids
                assert set(nodes)==set(source['selected_pair_indices'])
                for q in [q for q in questions if q['source_id']==source['source_id']]:
                    for arm in ARMS:
                        guard();chosen=choose(source,nodes,arm)
                        assert all(n.end<=q['query_time_s'] for n in chosen)
                        result=answer(env,model,builder,chosen,question(q,0,'B'))
                        result.update(parse(result['text'],len(q['options']),'B'))
                        result.update(id=q['id'],source_id=source['source_id'],role=q['type'],arm=arm,
                            query_time_s=q['query_time_s'],selected_pair_indices=[n.first for n in chosen])
                        output.write(json.dumps(result,ensure_ascii=False)+'\n');output.flush()
                        counts['answers']+=1;counts['tokens']+=len(result['generated_ids'])
                nodes.clear();gc.collect();torch.cuda.empty_cache();counts['sources']+=1
                print(json.dumps(dict(**counts,seconds=time.monotonic()-start)),flush=True)
        assert counts['answers']==88 and counts['warmup_answers']==4 and counts['vision']==159 and counts['sources']==11
        assert counts['tokens']<=plan['maximum_generated_tokens']
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        verify_lock()
        save(args.out/'completed.json',dict(status='completed',**counts,seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),updates=0,API=0,
            interpretation=plan['interpretation']))
    except BaseException as e:
        save(args.out/'failed.json',dict(error=repr(e),**counts));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
