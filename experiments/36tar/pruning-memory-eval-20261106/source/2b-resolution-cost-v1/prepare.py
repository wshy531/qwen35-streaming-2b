"""Freeze all existing formal questions plus a metadata-only independent cost panel."""
from collections import Counter,defaultdict
import hashlib
from resolution_common import ROOT,OUT,MAIN,SPECS,ARMS,read,save,sha


def main():
    inputs=OUT/'inputs';inputs.mkdir(parents=True,exist_ok=False)
    videos=read(MAIN/'inputs/videos.json');requests=read(MAIN/'inputs/requests.json')
    expected=read(MAIN/'inputs/expected_prefixes.json');byvideo=defaultdict(list)
    for q in requests:byvideo[q['video_id']].append(q)
    rank=lambda v:hashlib.sha256(('resolution-cost-20261025:'+v['video_id']).encode()).hexdigest()
    # Four sources in each duration stratum; greedy category coverage uses metadata only.
    panel=[];covered=set()
    for low,high in ((300,480),(480,720),(720,float('inf'))):
        pool=[v for v in videos if low<=v['duration_s']<high]
        for _ in range(4):
            pick=min(pool,key=lambda v:(-len({q['question_category'] for q in byvideo[v['video_id']]}-covered),rank(v)))
            panel.append(pick);pool.remove(pick);covered.update(q['question_category'] for q in byvideo[pick['video_id']])
    assert len(panel)==12 and len(covered)==8
    blocks=[sorted((v['video_id'] for v in panel[i::3])) for i in range(3)]
    assert len(set(v for b in blocks for v in b))==12
    totals=lambda vs:sum(expected[max(byvideo[v],key=lambda q:(q['question_time'],q['question_id']))['question_id']]['pairs'] for v in vs)
    panel_ids={v['video_id'] for v in panel};panel_q=sum(q['video_id'] in panel_ids for q in requests)
    pairs=totals([v['video_id'] for v in videos]);panel_pairs=totals(panel_ids)
    assert pairs==14678 and len(requests)==478 and len(videos)==128 and len(ARMS)==13
    warm=read(MAIN/'inputs/protocol.json')['warmup']
    for name,value in [('requests.json',requests),('videos.json',videos),('expected_mid_prefixes.json',expected)]:save(inputs/name,value)
    protocol=dict(stage='one_fixed_two_resolution_quality_and_independent_cost_comparison',
        questions=478,sources=128,arms=SPECS,quality_answers=478*13,
        cost_panel_sources=12,cost_panel_questions=panel_q,cost_blocks=blocks,
        cost_views=['LS_mid108','LS_highchange108','LS_high192'],cost_answers=panel_q*3,
        quality_processes=2,cost_processes=9,pretrained_loads=11,warmup_answers=22,
        max_answers=478*13+panel_q*3+22,max_generated_tokens=(478*13+panel_q*3+22)*16,
        pairs_per_resolution=pairs,cost_panel_pairs=panel_pairs,
        max_vision_encodings=2*pairs+3*panel_pairs+11,warmup=warm,
        dtype='FP32',sampling_FPS=.5,max_new_tokens=16,prefill_chunk=512,
        canvas_order=[[288,384],[384,512]],decoder='auto4',
        same_token_comparison='highchange108 versus mid108: both 864 bank tokens and same per-root size',
        expanded_budget_comparison='high192 versus mid108: same pair-equivalent capacities, 1536 versus 864 tokens',
        simple_control='M0_highuniform108 versus M0_highchange108, identical input and token budgets',
        source_identity_keys=['pairs','real_frames','pending_real_frames','last_committed_pts','true_EOF_tail_padded'],
        high_RGB_hash_not_equal_to_mid=True,mid_required_replay=['M0','M2','A06','LS'],
        prior_job='218526',prior_job_must_be_complete=True,
        bootstrap_source_resamples=5000,bootstrap_seed=20261025,
        decision=dict(no_automatic_default=True,one_round_no_ratio_sweep=True,
            admissible_cost_tradeoff=dict(all478_no_net_loss=True,old85_no_net_loss=True,
                at_least_one_group_net_gain=True,independent_LS_walltime_ratio_max=1.20,
                independent_peak_GPU_ratio_max=1.25),
            report_all_frontiers_even_if_gate_fails=True,small_positive_deltas_not_significance=True),
        GPU_authorized=False,training=False,download=False,automatic_retry=False,
        resource_budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,RAM_GiB=24,minutes=240),
        worker_RSS_limit_bytes=20*2**30,GPU_allocated_limit_bytes=48*2**30,log_limit_bytes=192*2**20,
        formal_primary_denominator=478,labels_on_worker=False,no_new_data=True,
        not_native_full_history_resolution_comparison=True,not_wallclock_1x_or_edge_measurement=True)
    save(inputs/'protocol.json',protocol)
    save(OUT/'selection_receipt.json',dict(status='frozen_before_new_outputs',
        inputs_sha={p.name:sha(p) for p in inputs.iterdir()},cost_panel=panel,
        cost_categories=sorted(covered),source_selection_independent_of_predictions=True))
    print({k:protocol[k] for k in ('quality_answers','cost_panel_questions','cost_answers','max_answers','max_generated_tokens','max_vision_encodings','pretrained_loads')})


if __name__=='__main__':main()
