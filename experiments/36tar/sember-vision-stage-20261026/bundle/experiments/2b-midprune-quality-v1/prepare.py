"""Reuse every formal question and the existing cost-panel membership; no selection on new scores."""
from quality_common import ROOT,OUT,MAIN,RESOLUTION,ARMS,read,save,sha,golden_path


def main():
    inputs=OUT/'inputs';inputs.mkdir(parents=True,exist_ok=False)
    requests=read(MAIN/'inputs/requests.json');videos=read(MAIN/'inputs/videos.json')
    expected=read(MAIN/'inputs/expected_prefixes.json');prior=read(RESOLUTION/'inputs/protocol.json')
    panel=read(RESOLUTION/'selection_receipt.json')['cost_panel']
    blocks=[sorted(v['video_id'] for v in panel[i::4]) for i in range(4)]
    assert len(videos)==128 and len(requests)==478 and len({v for b in blocks for v in b})==12
    panel_ids={v for b in blocks for v in b};panel_q=sum(q['video_id'] in panel_ids for q in requests)
    assert panel_q==57 and prior['pairs_per_resolution']==14678 and prior['cost_panel_pairs']==1910
    tasks=[dict(name='quality',kind='quality',arms=list(ARMS),videos=None)]
    for i,block in enumerate(blocks):
        for arm in ARMS[i:]+ARMS[:i]:tasks.append(dict(name=f'cost_b{i}_{arm}',kind='cost',arms=[arm],videos=block))
    warm=read(MAIN/'inputs/protocol.json')['warmup']
    protocol=dict(stage='R2_fixed_midpoint_odd_quality_and_independent_cost',
        sources=128,questions=478,canvas=[288,384],sampling_FPS=.5,decoder='auto4',dtype='FP32',
        arms=list(ARMS),pairs_per_source_limit=8,visual_budgets={a:864 if a==ARMS[0] else 432 for a in ARMS},
        fixed_cut_after_layers=12,fixed_keep_groups=54,original_spatial_coordinates=True,
        quality_answers=1912,cost_answers=228,warmup_answers=20,max_answers=2160,
        max_generated_tokens=34560,max_new_tokens=16,prefill_chunk=512,
        quality_vision_calls=2*14678,cost_vision_calls=4*1910,warmup_vision_calls=18,
        max_vision_encodings=37014,pretrained_loads=17,tasks=len(tasks),warmup=warm,
        cost_panel_sources=12,cost_panel_questions=57,cost_blocks=blocks,
        cost_order_balanced_four_blocks=True,cost_blocks_disjoint_not_repeated_videos=True,
        source_identity_keys=['pairs','real_frames','pending_real_frames','last_committed_pts',
            'true_EOF_tail_padded','submitted_prefix_sha256'],
        golden_job='218526',golden_rows=478*3,golden_answers_SHA=sha(golden_path()),labels_on_worker=False,
        decision=dict(all478_net_loss_max=4,old85_net_loss_max=0,independent_total_wall_ratio_max=.95,
            followup_not_statistical_noninferiority=True,no_automatic_default=True,
            no_layer_or_ratio_sweep=True,report_wins_losses_and_frontier_even_if_gate_fails=True),
        resource_budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,RAM_GiB=24,minutes=120),
        GPU_authorized=False,training=False,download=False,automatic_retry=False,requeue=False,
        worker_RSS_limit_bytes=20*2**30,GPU_allocated_limit_bytes=32*2**30,log_limit_bytes=128*2**20,
        load_warmup_and_media_SHA_outside_cost=True,quality_shared_vision_not_E2E=True,
        standalone_cost_includes_decode_canvas_processor_vision_update_QA_and_audits=True,
        R1_gate_still_failed=True,R2_user_accepts_actual_14percent_tradeoff=True,
        not_edge_or_wallclock_1x=True,all_questions_opened_development=True)
    assert len(tasks)==17
    for name,value in [('requests.json',requests),('videos.json',videos),('expected_prefixes.json',expected),
        ('tasks.json',tasks),('protocol.json',protocol)]:save(inputs/name,value)
    labels=ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'
    save(OUT/'local_analysis_identity.json',dict(labels_SHA=sha(labels),labels_not_deployed=True))
    save(OUT/'selection_receipt.json',dict(status='frozen_before_R2_outputs',
        all478_reused=True,cost_panel_membership_unchanged=True,cost_panel_source_sha=sha(RESOLUTION/'selection_receipt.json'),
        cost_blocks=blocks,no_new_predictions_used=True))
    print({k:protocol[k] for k in ('max_answers','max_generated_tokens','max_vision_encodings','pretrained_loads')},flush=True)


if __name__=='__main__':main()
