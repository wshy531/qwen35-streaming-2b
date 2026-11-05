"""All 345 existing development questions, original protocol and endpoint controls."""
import sys,os,json,time,gc,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/anchor_merge_development_20261103_01'
sys.path.insert(0,str(R/'experiments/2b-learned-prune-main-v1'))
import protocol as old
spec=importlib.util.spec_from_file_location('anchor_original_qa',R/'experiments/2b-learned-prune-main-v1/worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from aligned_encoder import encode
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def verify():
    for name,h in read(P/'execution_lock.json').items():
        p=R/name
        if name.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/name[5:]
        assert old.sha(p)==h,name
def main(out):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm required')
    verify();auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['execution_lock_sha256']==old.sha(P/'execution_lock.json')
    assert not out.exists() and out.resolve().is_relative_to((R/'runs').resolve());out.mkdir()
    qs=read(P/'questions.json');sources=read(P/'sources.json');audits={s['video_id']:s for s in read(P/'samples.json')}
    refs={(x['id'],x['arm'],x['shift']):x for x in read(P/'baseline.json')};plan=read(P/'protocol.json')
    env=old.runtime();t=env['torch'];t.set_num_threads(1)
    assert t.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
    start=time.monotonic();answers=generated=vision=controls=warmups=0
    def guard():
        if time.monotonic()-start>58*60:raise TimeoutError('fixed 58-minute stop')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<96*2**20
    try:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=old.make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for si,source in enumerate(sorted(sources,key=lambda s:s['video_id'])):
                guard();vid=source['video_id'];path=R/'datasets/learned-prune-nextqa-v1'/source['mirror_path']
                assert old.sha(path)==source['mirror_lfs_sha256']
                pairs=old.decode(env,path,audits[vid],builder);questions=[q for q in qs if q['video_id']==vid]
                for arm in plan['arms']:
                    selector=old.CoverageScorer();records=[]
                    with env['fp32_reference']():
                        for i,(begin,end,batch) in enumerate(pairs):
                            guard();batch=batch.to('cuda')
                            if arm=='anchor_merge11':feat,ids,_=encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,selector)
                            else:feat,ids,_=old.encode_pair(model,batch,arm,None,None,selector)
                            records.append((i,begin,end,feat,ids));vision+=1
                        if si==0:
                            r=worker.answer(env,model,builder,records,old.rotate(questions[0],0));warmups+=1;generated+=len(r['generated_ids'])
                        for q in questions:
                            for shift in plan['rotations']:
                                guard();r=worker.answer(env,model,builder,records,old.rotate(q,shift));answers+=1;generated+=len(r['generated_ids'])
                                row=dict(r,id=q['id'],video_id=vid,arm=arm,shift=shift,type=q['type'])
                                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                                if arm!='anchor_merge11':
                                    ref=refs[q['id'],arm,shift]
                                    for key in ('generated_ids','text','input_ids_sha256','input_tokens'):assert r[key]==ref[key],(q['id'],arm,key)
                                    controls+=1
                    del records;gc.collect()
                del pairs;t.cuda.empty_cache();print(json.dumps(dict(sources=si+1,answers=answers,controls=controls)),flush=True)
        assert (answers,controls,warmups,vision)==(2760,2070,4,plan['vision_calls']) and generated<=22112
        verify();save(out/'completed.json',dict(status='complete',answers=answers,controls=controls,warmups=warmups,generated_tokens=generated,vision_calls=vision,seconds=time.monotonic()-start,training=0,pretrained_loads=1,not_standalone_cost=True))
    except Exception as e:save(out/'failed.json',dict(error=str(e),answers=answers,controls=controls));raise
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
