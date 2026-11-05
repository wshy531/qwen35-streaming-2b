"""Lock a model-independent curated comparison roster; preserve gate separation."""
from collections import Counter
import csv
import statistics

from evaluation_gate import official_identity,freeze_issues
from prefix_request import make_request
from prepare_natural_v2 import ROOT,OUT,BASE,DATA,read,save,sha
from score_memory_eval import digest,prompt

EXCLUDE={
 'nextgqa:val:6618792571:9':'在实际模型画布中成人如何介入不清楚，不能以镜头靠近代替成人帮助证据。',
 'nextgqa:val:6618792571:10':'实际画布不能可靠辨认成人抓握及拉动车把，保留主张而不纳入主比较。',
 'nextgqa:val:4276006683:7':'沿原复核已经记录的选项重叠：bite和grab hand部分同时成立，仅作补充。',
}


def main():
    rows=read(OUT/'all149_dispositions.json');candidates=read(OUT/'candidate77.json')
    canvas=read(OUT/'model_canvas_sheets/receipts.json')
    images={im['path']:sha(ROOT/im['path']) for source in canvas['sources'] for im in source['images']}
    assert len(images)==24
    frame_map={s['source_path']:{f['frame_id']:f for f in s['frames']} for s in canvas['sources']}
    assert set(q['id'] for q in candidates)>set(EXCLUDE)
    included=[];allrows=[]
    for q in rows:
        q=dict(q);q['admission']=dict(include_in_comparison=q['curation']['candidate'] and q['id'] not in EXCLUDE,
            exclusion_reason=EXCLUDE.get(q['id']) or (None if q['curation']['candidate'] else q['curation']['note']),
            independent_human_review=False,comparison_model_outputs_seen=False,
            model_canvas_reviewed=q['curation']['candidate'])
        if q['admission']['include_in_comparison']:
            for w in q['curation']['witnesses']:
                f=frame_map[q['source_path']][w['frame_id']]
                assert f['rgb_sha256']==w['rgb_sha256'] and f['pts_s']<=q['query_time_s']
            q['review']=dict(verdict='supported_interpretation' if q['curation']['stratum']=='C' else 'supported_observation',
                reviewer='Codex single-agent visual curation; initial and exact model-canvas passes; not independent human review',
                note=q['curation']['note'],flags_pending=[],model_outputs_seen=False,witnesses=q['curation']['witnesses'],
                input_resolution_hw=[192,256],sampling_fps=1.,stratum=q['curation']['stratum'],
                physical_causal_ground_truth_claimed=False)
            included.append(q)
        allrows.append(q)
    assert len(included)==74 and len(allrows)==149
    traces={t['source_path']:t for t in read(OUT/'sampling_1fps.json')}
    requests=[];strata=[]
    for q in included:
        request=make_request(q,dict(source_path=q['source_path'],media_path=q['media_path'],sha256=q['media_sha256']),traces[q['source_path']])
        request['input_protocol']='natural-v2_1fps_192x256_data_locked_execution_pending';requests.append(request)
        frames=request['model_visible']['frames'];groups=[frames[i:i+2] for i in range(0,len(frames),2)]
        tail=[f for group in groups[-8:] for f in group]
        intervals=q['evidence_intervals'];fullhits=[any(a<=f['pts_s']<=b for f in frames) for a,b in intervals]
        recenthits=[any(a<=f['pts_s']<=b for f in tail) for a,b in intervals]
        if len(groups)<=8:pressure='no_8pair_capacity_pressure'
        elif not all(fullhits):pressure='full_sampling_misses_some_author_intervals'
        elif not any(recenthits):pressure='recent_misses_all_author_intervals'
        elif not all(recenthits):pressure='recent_misses_some_author_intervals'
        else:pressure='recent_covers_all_author_intervals'
        gap=max(0,q['query_time_s']-max(b for a,b in intervals)) if intervals else None
        age='unknown' if gap is None else '0-10s' if gap<=10 else '10-30s' if gap<=30 else '30-60s' if gap<=60 else '>60s'
        strata.append(dict(id=q['id'],source_group_id=q['source_group_id'],stratum=q['curation']['stratum'],
            official_task=q['task'],query_time_s=q['query_time_s'],real_frames=len(frames),pairs=len(groups),
            author_interval_hits_full=fullhits,author_interval_hits_recent=recenthits,evidence_gap_proxy_s=gap,
            evidence_age_bin=age,pressure_proxy=pressure,requires_old_evidence_proven=False,
            note='Author ranges are supporting annotations, not proof of necessary or exclusive evidence.'))
    save(OUT/'admitted74.json',included);save(OUT/'all149_final_dispositions.json',allrows)
    save(OUT/'model_requests74.json',requests);save(OUT/'strata74.json',strata)
    key=[dict(id=q['id'],answer_index=q['answer_index'],stratum=q['curation']['stratum'],source_group_id=q['source_group_id']) for q in included]
    save(OUT/'scorer_only_key74.json',key)
    registry=[dict(source_group_id=s['source_group_id'],split=s['split']) for s in read(BASE/'content_release_v1/source_registry.json')['sources'] if s['source_group_id']]
    originals=[official_identity(q) for q in read(OUT/'original149.json')]
    protocol=dict(sampling_fps=1.,canvas_hw=[192,256],pairing_policy='two_real_or_snapshot_only_last_duplicate',
        query_cutoff_policy='pts_not_after_query',readout_layout='existing external-memory chronological labels and destination coordinate heuristic',
        generation=dict(max_new_tokens=8,do_sample=False,enable_thinking=False),visual_token_budget=384,
        memory_budget_scope='0.8B actual P=48; M0 8P, M1/M2 4P recent plus4P long, M2 openP plusclosed3P',
        full_reference_limits=dict(context_tokens=8192,visual_tokens=6144,nodes=128,suffix_tokens=128),
        input_trace_sha256=digest([q['model_visible'] for q in requests]),prompt_template_sha256=digest([prompt(q) for q in included]),
        model_revision=None,processor_revision=None,precision='float32',real_interface_receipt_sha256=None,
        no_answer_or_author_interval_in_model_visible=True,real_1fps_outputs=0)
    bundle=dict(status='data_roster_locked_execution_admission_pending',schema='2b-memory-eval-v1',evaluation_scope='comparison',
        rows=included,original_rows=originals,original_rows_sha256=digest(originals),
        split_registry=registry,split_registry_sha256=digest(registry),input_protocol=protocol)
    save(OUT/'execution_bundle_draft.json',bundle)
    issues=freeze_issues(bundle)
    assert 'bundle_not_frozen' in issues
    assert not any('original_question_or_label_changed' in i or 'source_recording_leaks' in i for i in issues)
    ids={q['source_group_id'] for q in included};durations=[next(q['query_time_s'] for q in included if q['source_group_id']==i) for i in ids]
    summary=dict(status='data_roster_and_analysis_rules_locked_no_model_test',source_scope=24,original_questions=149,
        comparison_videos=len(ids),comparison_questions=74,not_in_main_comparison=75,
        actual_observation_questions=sum(q['curation']['stratum'] in ['F','T'] for q in included),
        contextual_explanation_questions=sum(q['curation']['stratum']=='C' for q in included),
        stratum_counts=dict(Counter(q['curation']['stratum'] for q in included)),
        tasks=dict(Counter(q['task'] for q in included)),
        video_query_duration_s=dict(min=min(durations),median=statistics.median(durations),max=max(durations)),
        video_length_bins=dict(under30=sum(d<30 for d in durations),from30to60=sum(30<=d<60 for d in durations),from60to180=sum(60<=d<180 for d in durations),over180=sum(d>=180 for d in durations)),
        evidence_age_proxy_counts=dict(Counter(s['evidence_age_bin'] for s in strata)),
        memory_pressure_proxy_counts=dict(Counter(s['pressure_proxy'] for s in strata)),
        questions_per_video=dict(Counter(q['source_group_id'] for q in included)),
        inspected_native_canvas_images=images,canvas_exclusions=EXCLUDE,holdout_images_seen=0,models_loaded=0,
        formal_model_bundle_frozen=False,execution_admission_issues=issues,
        limitations=['Single-agent annotation, no independent human adjudication.',
            'Contextual explanations reported separately; not ground truth of mental states or intervention causality.',
            'Short natural clips only: this roster does not test multi-minute long-term memory.',
            'Author-range and capacity pressure are proxies; strict dependence on old evidence still requires controls.',
            'Related questions are clustered by canonical video; more questions are not more independent recordings.',
            'Official public content may have been in pretraining; benchmark contamination cannot be excluded.',
            'Data content lock is distinct from a model-specific execution gate and GPU authorization.'])
    save(OUT/'summary.json',summary)
    rules=dict(status='locked_before_comparison_outputs',main_population='all admitted74, no post-hoc filtering by reference correctness',
        primary='Report video-macro accuracy separately for F/T observations and C contextual explanations; F, T, C task tables always shown.',
        secondary='Micro accuracy, full74 labeled aggregate, strict-format failures and all error statuses; show all denominators.',
        scoring='Existing score_memory_eval.parse_choice strict single A-E letter; missing/format/timeout/OOM count wrong in full population.',
        paired='Both-correct, correction, new-loss, both-wrong; complete reference failures retained; reference-correct subset supplementary only.',
        uncertainty='Resample canonical video groups as clusters, same resampled groups across arms, fixed seed20261019,2000 bootstrap replicates,95percent percentile intervals; no independence claim for individual questions.',
        predeclared_strata=['F/T/C','official_task','video_length','evidence_age_proxy','memory_pressure_proxy'],
        controls=['original_native','full_pairwise_reference','M0_recent8P','M1_complete_representatives','M2_auto_forest','text_only'],
        diagnostic_only='Fixed4 grouping may be a separately preregistered appendix; not added after seeing comparison failures.',
        no_online_frame_dropping=True,no_training=True,no_prompt_threshold_or_budget_tuning_on_comparison=True,
        next_candidate='CausalMem only after this frozen W1 comparison; no extra new architecture branch.',
        holdout='Original40 files/94 questions remain unviewed; qualify independently after method settings fixed, before final holdout test.')
    save(OUT/'analysis_rules.json',rules)
    with (OUT/'主比较74题清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','source_group_id','question','official_answer','stratum','query_time_s','evidence_age_bin','pressure_proxy','note'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        smap={q['id']:q for q in strata}
        for q in included:
            sm=smap[q['id']];writer.writerow(dict(id=q['id'],source_group_id=q['source_group_id'],question=q['question'],official_answer=q['options'][q['answer_index']],
                stratum=q['curation']['stratum'],query_time_s=q['query_time_s'],evidence_age_bin=sm['evidence_age_bin'],pressure_proxy=sm['pressure_proxy'],note=q['curation']['note']))
    print({k:v for k,v in summary.items() if k not in ['inspected_native_canvas_images','questions_per_video','limitations','execution_admission_issues','canvas_exclusions']})
    print('execution gate issues:',issues)


if __name__=='__main__':main()
