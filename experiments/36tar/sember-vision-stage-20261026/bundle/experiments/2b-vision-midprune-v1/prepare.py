"""Select fixed development sources/positions without reading questions or scores."""
import hashlib
from common import ROOT,OUT,read,save


def main():
    source=ROOT/'runs/sember_prepare_20261025_01/selection/videos.json'
    videos=read(source)
    candidates=[v for v in videos if v['split']=='development18']
    assert len(candidates)==18
    ordered=sorted(candidates,key=lambda v:hashlib.sha256(('midprune-v1:'+v['video_id']).encode()).hexdigest())
    selected=ordered[:4]
    pairs=[]
    for v in selected:
        for fraction in (.1,.5,.9):
            index=int(v['duration_s']*fraction//4)
            end=index*4+2
            assert 0<end<v['duration_s']
            pairs.append(dict(video_id=v['video_id'],fraction=fraction,pair_index=index,
                query_time=end,first_sample_time=index*4,second_sample_time=end))
    (OUT/'inputs').mkdir(parents=True)
    save(OUT/'inputs/videos.json',selected)
    save(OUT/'inputs/pairs.json',pairs)
    save(OUT/'inputs/protocol.json',dict(hypothesis='fixed group pruning moved inside ViT reduces actual visual cost',
        canvas=[288,384],groups_full=108,groups_retained=54,prune_after_layers=12,visual_depth=24,
        modes=['original_full','post_odd','mid_odd'],pair_count=12,source_count=4,
        warmup_repeats=5,measured_repeats=20,control_forwards=24,
        max_vision_forwards=924,QA_answers=0,generated_tokens=0,pretrained_loads=1,
        source_selection='development18 video SHA order; first four',
        pair_positions='10/50/90 percent of source duration, aligned to original causal pair grid',
        selected_without_labels_or_answers=True,frame_cache_bytes_limit=8*2**20,
        benchmarking_preloaded_frames_not_end_to_end=True,no_capability_claim_from_R1=True,
        gate=dict(median_pair_time_ratio_max=.85,max_pairs_slower_than_full=3,
            all_kept_gpu_max_abs_error=1e-5,all_kept_gpu_rtol=1e-5),
        timing='CUDA synchronize around each complete vision+selection call; rotation balances order',
        budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,RAM_GiB=24,minutes=15),
        log_limit_bytes=16*2**20,worker_RSS_limit_bytes=20*2**30,GPU_allocated_limit_bytes=32*2**30,
        no_layer_or_ratio_sweep=True,training=False,download=False,requeue=False,automatic_retry=False,
        GPU_authorized=False))
    print(dict(status='fixed_real_pairs_selected',sources=4,pairs=12,answers_read=0),flush=True)


if __name__=='__main__':main()
