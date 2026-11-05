"""Five-mode GPU component-cost precheck. Fixed old 12 pairs, no QA claims."""
import os,sys,json,time,gc,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PREP=ROOT/'runs/content_midprune_prepare_20261101_01'
sys.path.insert(0,str(ROOT/'experiments/2b-vision-midprune-v1'))
from common import runtime,read
from source_pairs import load_pairs,media_root
from content_prune import encode,CoverageScorer
MODES=['full','post_fixed','mid_fixed','post_content','mid_content']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    lock=read(PREP/'execution_lock.json')
    for rel,h in lock.items():
        p=ROOT/rel
        if rel.startswith('repo/') and not p.exists():p=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(p)==h,rel
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)
def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm allocation required')
    verify();a=read(PREP/'GPU_authorization.json')
    assert a['approved'] and a['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    env=runtime();t=env['torch'];t.set_num_threads(1)
    assert t.cuda.device_count()==1 and t.cuda.mem_get_info()[0]>=24*2**30
    out=ROOT/'runs'/('content-midprune-cost-'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    started=time.monotonic();calls=0;rows=[]
    try:
        old=ROOT/'runs/vision_midprune_prepare_20261025_01/inputs'
        for v in read(old/'videos.json'):assert sha(media_root()/v['relative_path'])==v['sha256']
        model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=t.float32,device_map='cuda',attn_implementation='sdpa').float().eval()
        model.requires_grad_(False);visual=model.model.visual;assert len(visual.blocks)==24
        processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
        pairs,rgbbytes=load_pairs();assert rgbbytes<8*2**20
        def guard():
            assert time.monotonic()-started<780 and env['psutil'].Process().memory_info().rss<20*2**30
            assert t.cuda.max_memory_allocated()<32*2**30 and calls<=960
        import numpy as np
        with t.inference_mode(),env['fp32_reference']():
            for serial,item in enumerate(pairs):
                guard();inputs=processor.video_processor(videos=[np.stack([item['first'].rgb,item['second'].rgb])],
                    do_resize=False,do_sample_frames=False,return_metadata=False,return_tensors='pt').to('cuda')
                x,g=inputs.pixel_values_videos,inputs.video_grid_thw
                official=visual(x,grid_thw=g).pooler_output;calls+=1
                full,_,_=encode(visual,x,g,mode='full');calls+=1
                assert t.allclose(full,official,atol=1e-5,rtol=1e-5)
                # Prime independent scorers with the same full middle-layer descriptor.
                # Repeated benchmark uses a stationary previous pair: cost only, not temporal quality.
                states={m:CoverageScorer() for m in ('post_content','mid_content')}
                for m,s in states.items():encode(visual,x,g,mode=m,scorer=s);calls+=1
                snapshots={m:s.previous.clone() for m,s in states.items()}
                timings={m:[] for m in MODES};ids_seen={};extra={}
                for rep in range(15):
                    order=MODES[(serial+rep)%5:]+MODES[:(serial+rep)%5]
                    for mode in order:
                        scorer=states.get(mode)
                        if scorer is not None:scorer.previous=snapshots[mode]  # restore outside measured interval
                        t.cuda.synchronize();tick=time.monotonic()
                        if mode=='full':value=visual(x,grid_thw=g).pooler_output;ids=t.arange(108,device=x.device)
                        else:value,ids,_=encode(visual,x,g,mode=mode,scorer=scorer)
                        t.cuda.synchronize();elapsed=time.monotonic()-tick;calls+=1
                        assert t.isfinite(value).all()
                        ids_seen[mode]=ids.cpu().tolist();extra[mode]=0 if scorer is None else scorer.resident_bytes()
                        if rep>=5:timings[mode].append(elapsed)
                    guard()
                assert ids_seen['post_content']==ids_seen['mid_content']
                med={m:statistics.median(v) for m,v in timings.items()}
                rows.append(dict(pair=item['spec'],medians_seconds=med,samples_seconds=timings,
                    extra_selector_bytes=extra,mid_content_vs_full=med['mid_content']/med['full'],
                    mid_content_vs_fixed=med['mid_content']/med['mid_fixed']))
                save(out/f'pair_{serial:02d}.json',rows[-1]);del inputs,x,g,official,full,value,states,snapshots
                gc.collect();t.cuda.empty_cache()
        assert calls==948
        verify();save(out/'completed.json',dict(status='complete',vision_calls=calls,rows=rows,pretrained_loads=1,
            wall_seconds=time.monotonic()-started,RGB_bytes=rgbbytes,QA_answers=0,generated_tokens=0,
            peak_GPU_allocated_bytes=t.cuda.max_memory_allocated(),stationary_cache_timing=True,
            not_temporal_quality=True,not_end_to_end=True,not_edge=True))
    except Exception as e:save(out/'failed.json',dict(error=str(e),vision_calls=calls));raise
if __name__=='__main__':main()
