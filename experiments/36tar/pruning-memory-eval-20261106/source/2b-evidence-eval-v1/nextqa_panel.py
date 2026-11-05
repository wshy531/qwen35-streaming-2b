"""Frozen 30-question development panel, three methods, two protocols/rotations."""
import argparse,gc,json,os,time
from pathlib import Path
import gpu_profile as shared
from response_v3 import parse
R=shared.R;P=shared.P
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for n,h in shared.read(P/'nextqa_panel_lock.json').items():assert shared.sha(R/n)==h,n
    assert not a.out.exists();a.out.mkdir();start=time.monotonic();answers=calls=tokens=0
    try:
        env=shared.runtime();t=env['torch'];t.set_num_threads(1);assert t.cuda.device_count()==1
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,
            device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=shared.make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        audit={r['video_id']:r for r in shared.read(shared.DATA/'CPU_media_audit.json')['rows']}
        qs=[q for q in shared.read(P/'panel.json') if q['dataset']=='NExT-QA'];assert len(qs)==30
        for q in qs:assert not {'answer_index','answer_text'}.intersection(q)
        arms=('full','fixed12','heuristic11');vids=sorted({q['source_id'] for q in qs});checked=[]
        with t.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for vi,vid in enumerate(vids):
                q=next(q for q in qs if q['source_id']==vid);path=R/q['media_path'];assert shared.sha(path)==q['media_sha256']
                pairs=shared.decode(env,path,audit[vid],builder);checked.append(dict(video_id=vid,sha256=q['media_sha256'],pairs=len(pairs)))
                for arm in arms[vi%3:]+arms[:vi%3]:
                    records=shared.records_for(model,pairs,arm);calls+=len(pairs)
                    for q in [q for q in qs if q['source_id']==vid]:
                        for shift in (0,2):
                            rq=dict(q,options=q['options'][shift:]+q['options'][:shift])
                            for protocol in ('A','B'):
                                if time.monotonic()-start>27*60:raise TimeoutError('panel deadline')
                                result=shared.generate(env,model,builder,records,rq,protocol)
                                result.update(parse(result['text'],len(q['options']),protocol,shift))
                                result.update(id=q['id'],source_id=vid,type=q['type'],arm=arm,protocol=protocol,shift=shift)
                                answers+=1;tokens+=len(result['generated_ids']);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                    del records;gc.collect()
                del pairs;t.cuda.empty_cache();print(json.dumps(dict(sources=vi+1,answers=answers,seconds=time.monotonic()-start)),flush=True)
        assert answers==360 and tokens<=24480
        shared.save(a.out/'completed.json',dict(status='complete_development_generation_only',answers=answers,vision_calls=calls,
            generated_tokens=tokens,model_loads=1,seconds=time.monotonic()-start,API=0,training=0,checked_media=checked,
            note='API evidence scores not yet run; no held-out capability claim'))
    except Exception as e:shared.save(a.out/'failed.json',dict(error=repr(e),answers=answers,vision_calls=calls));raise
if __name__=='__main__':main()
