"""Cost feasibility only: twelve real pairs, original/post/mid rules, no questions or generation."""
import gc
import os
from pathlib import Path
import statistics
import time
from common import ROOT,OUT,runtime,read,save,sha,verify
from midprune import encode
from source_pairs import media_root,load_pairs


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit new GPU approval and Slurm allocation required')
    protocol=verify();authorization=read(OUT/'GPU_authorization.json')
    assert authorization['approved'] and authorization['execution_lock_sha256']==sha(OUT/'inputs/execution_lock.json')
    job=os.environ['SLURM_JOB_ID'];assert job.isdigit()
    out=ROOT/'runs'/('2b-midprune-cost-'+job);out.mkdir(exist_ok=False)
    deadline=time.time()+15*60-90
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>=24*2**30
    for video in read(OUT/'inputs/videos.json'):
        path=(media_root()/video['relative_path']).resolve()
        assert path.is_relative_to(media_root().resolve()) and sha(path)==video['sha256']
    loaded=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    load_seconds=time.monotonic()-loaded
    visual=model.model.visual;assert len(visual.blocks)==24 and visual.config.out_hidden_size==2048
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    source_started=time.monotonic();pairs,frame_bytes=load_pairs();source_seconds=time.monotonic()-source_started
    save(out/'startup.json',dict(pretrained_loads=1,load_seconds=load_seconds,
        selected_pair_preload_seconds=source_seconds,selected_RGB_bytes=frame_bytes,
        source_count=4,pair_count=12,QA_answers=0,generated_tokens=0,
        input_preload_and_model_load_outside_visual_timer=True,
        does_not_measure_end_to_end_or_edge_speed=True))
    torch.cuda.reset_peak_memory_stats();counter=0;rows=[]
    def guard():
        assert time.time()<deadline,'approved walltime reached'
        assert env['psutil'].Process().memory_info().rss<protocol['worker_RSS_limit_bytes']
        assert torch.cuda.max_memory_allocated()<protocol['GPU_allocated_limit_bytes']
        assert counter<=protocol['max_vision_forwards']
    with torch.inference_mode(),env['fp32_reference']():
        for serial,item in enumerate(pairs):
            guard()
            import numpy as np
            pixels=processor.video_processor(videos=[np.stack([item['first'].rgb,item['second'].rgb])],
                do_resize=False,do_sample_frames=False,return_metadata=False,return_tensors='pt').to('cuda')
            x,grid=pixels.pixel_values_videos,pixels.video_grid_thw
            assert grid.tolist()==[[1,18,24]]
            original=visual(x,grid_thw=grid).pooler_output;counter+=1
            replay,ids,trace=encode(visual,x,grid,mode='full_replay');counter+=1
            error=float((original-replay).abs().max())
            assert error<=protocol['gate']['all_kept_gpu_max_abs_error']
            assert torch.allclose(original,replay,atol=1e-5,rtol=1e-5)
            ids=torch.arange(1,108,2,device=x.device)
            modes=protocol['modes'];samples={mode:[] for mode in modes};output_shapes={}
            for repeat in range(protocol['warmup_repeats']+protocol['measured_repeats']):
                order=modes[(repeat+serial)%3:]+modes[:(repeat+serial)%3]
                for mode in order:
                    torch.cuda.synchronize();started=time.monotonic()
                    if mode=='original_full':features=visual(x,grid_thw=grid).pooler_output
                    else:features,selected,trace=encode(visual,x,grid,mode=mode)
                    torch.cuda.synchronize();elapsed=time.monotonic()-started;counter+=1
                    assert torch.isfinite(features).all()
                    if mode=='post_odd':assert torch.allclose(features,original[ids],atol=1e-5,rtol=1e-5)
                    output_shapes[mode]=list(features.shape)
                    if repeat>=protocol['warmup_repeats']:samples[mode].append(elapsed)
                guard()
            medians={mode:statistics.median(v) for mode,v in samples.items()}
            rows.append(dict(pair=item['spec'],first=item['first'].record(),second=item['second'].record(),
                all_kept_max_abs_error=error,output_shapes=output_shapes,samples_seconds=samples,
                medians_seconds=medians,mid_vs_original_ratio=medians['mid_odd']/medians['original_full'],
                mid_vs_post_ratio=medians['mid_odd']/medians['post_odd']))
            del original,replay,pixels,x,grid,features
            gc.collect();torch.cuda.empty_cache()
            save(out/f'pair_{serial:02d}.json',rows[-1])
    assert counter==924
    save(out/'completed.json',dict(status='complete',pretrained_loads=1,vision_forwards=counter,
        pair_count=12,QA_answers=0,generated_tokens=0,rows=rows,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),last_RSS_bytes=env['psutil'].Process().memory_info().rss,
        no_cross_frame_feature_cache=True,does_not_measure_end_to_end=True,
        not_a_quality_or_memory_test=True,execution_lock_sha256=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':main()
