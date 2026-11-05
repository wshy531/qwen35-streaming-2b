"""30 frozen online queries: causal decoding and bounded last-eight-pair input."""
import argparse,collections,gc,json,os,sys,time
import gpu_profile as s
from response_v3 import parse
R=s.R;P=s.P
sys.path.insert(0,str(R/'experiments/streamingbench-interface-v1'))
from source import CausalPairs
ARMS=('full','fixed12','heuristic11')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=__import__('pathlib').Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for n,h in s.read(P/'stream_panel_lock.json').items():assert s.sha(R/n)==h,n
    assert not a.out.exists();a.out.mkdir();start=time.monotonic();answers=calls=tokens=0
    def guard():
        if time.monotonic()-start>37*60:raise TimeoutError('stream panel deadline')
        assert sum(x.stat().st_size for x in a.out.rglob('*') if x.is_file())<128*2**20
    try:
        env=s.runtime();t=env['torch'];t.set_num_threads(1);assert t.cuda.device_count()==1
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,
            device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=s.make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        qs=[q for q in s.read(P/'panel_media_bound.json') if q['dataset']=='StreamingBench'];assert len(qs)==30
        source_checks=[]
        with t.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for vi,vid in enumerate(sorted({q['source_id'] for q in qs})):
                q=next(q for q in qs if q['source_id']==vid);path=R/q['media_path'];assert s.sha(path)==q['media_sha256']
                banks={arm:collections.deque(maxlen=8) for arm in ARMS};selectors={arm:s.CoverageScorer() for arm in ARMS}
                pair_index=0;provenance=[]
                def commit(first,second,padding):
                    nonlocal pair_index,calls
                    guard();batch,_=builder.pair(first,second,0,0,synthetic_padding=padding);batch=batch.to('cuda')
                    for arm in ARMS:
                        features,ids,_=s.encode_pair(model,batch,arm,None,None,selectors[arm]);calls+=1
                        banks[arm].append((pair_index,first.timestamp,second.timestamp,features,ids))
                    provenance.append(dict(pair=pair_index,first_pts=first.timestamp,second_pts=second.timestamp,
                        first_frame=first.frame_id,second_frame=second.frame_id,padding=padding));pair_index+=1
                driver=CausalPairs(path,commit)
                try:
                    for q in sorted((q for q in qs if q['source_id']==vid),key=lambda q:(q['query_time_s'],q['id'])):
                        guard();state=driver.advance(q['query_time_s']);assert all(x[2]<=q['query_time_s']+1e-6 for x in banks['full'])
                        assert banks['full'],'no complete received pair'
                        for arm in ARMS[vi%3:]+ARMS[:vi%3]:
                            records=list(banks[arm]);assert len(records)<=8
                            for shift in (0,2):
                                rq=dict(q,options=q['options'][shift:]+q['options'][:shift])
                                for proto in ('A','B'):
                                    guard();result=s.generate(env,model,builder,records,rq,proto)
                                    result.update(parse(result['text'],len(q['options']),proto,shift))
                                    result.update(id=q['id'],source_id=vid,type=q['type'],arm=arm,protocol=proto,shift=shift,
                                        query_time_s=q['query_time_s'],visible_pairs=[x[0] for x in records],source_state=state)
                                    answers+=1;tokens+=len(result['generated_ids']);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                        print(json.dumps(dict(queries=answers//12,answers=answers,seconds=time.monotonic()-start)),flush=True)
                finally:driver.close()
                source_checks.append(dict(source_id=vid,media_sha256=q['media_sha256'],pairs=pair_index,pair_provenance=provenance))
                del banks,selectors,records;gc.collect();t.cuda.empty_cache()
        assert answers==360 and tokens<=24480
        s.save(a.out/'completed.json',dict(status='complete_online_development_generation_only',answers=answers,vision_calls=calls,
            generated_tokens=tokens,model_loads=1,seconds=time.monotonic()-start,API=0,training=0,
            checked_sources=source_checks,capacity_visual_tokens=dict(full=864,fixed12=432,heuristic11=432),
            not_wallclock_1x=True,question_branch_states_released=True))
    except Exception as e:s.save(a.out/'failed.json',dict(error=repr(e),answers=answers,vision_calls=calls));raise
if __name__=='__main__':main()
