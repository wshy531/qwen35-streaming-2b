"""Bind all94 locked holdout dispositions to actual RGB witnesses, no inference.

The single-review data roster is not a model-specific execution admission.
Original labels, options, splits and all excluded rows remain in the ledger.
"""
from collections import Counter
import csv
import statistics

from prepare_holdout_review import ROOT,BASE,OUT,DATA,read,save,sha
from evaluation_gate import official_identity,freeze_issues
from prefix_request import make_request
from score_memory_eval import digest,prompt


def counts(rows):
    sources={q['source_group_id']:q['query_time_s'] for q in rows}
    durations=list(sources.values())
    return dict(questions=len(rows),sources=len(sources),strata=dict(Counter(q['curation']['stratum'] for q in rows)),
        tasks=dict(Counter(q['task'] for q in rows)),
        query_duration_s=dict(min=min(durations),median=statistics.median(durations),max=max(durations)) if durations else None,
        length_bins=dict(under30=sum(t<30 for t in durations),from30to60=sum(30<=t<60 for t in durations),
            from60to180=sum(60<=t<180 for t in durations),over180=sum(t>=180 for t in durations)))


def main():
    import json
    plan=read(OUT/'annotation_plan.json');original=read(OUT/'original94.json')
    assert sha(DATA/'candidate_manifest.json')=='cd1d888eb0a0c00f300649f41800e60fd02b002e91f41f154b42e041fcc28161'
    assert sha(OUT/'original94.json')==plan['original94_sha256']
    assert sha(OUT/'previsual_metadata94.json')==plan['metadata_sha256']
    assert original==[q for q in read(DATA/'candidate_manifest.json') if q['split']=='holdout']
    assert len(original)==94 and len({q['source_path'] for q in original})==40
    for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
    meta=read(OUT/'previsual_metadata94.json')
    decisions=[json.loads(s) for s in (OUT/'visual_decisions.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    dm={d['id']:d for d in decisions};assert len(dm)==len(decisions)==76
    assert set(dm)=={m['original']['id'] for m in meta if m['image_review_planned']}
    receipts=read(OUT/'image_receipts.json');images={}
    receipt_map={r['source_path']:r for r in receipts}
    traces={r['source_path']:r for r in read(OUT/'sampling_1fps.json')}
    for r in receipts:
        assert len(r['witnesses'])<=24 and len(r['images'])<=2
        for im in r['images']:
            assert sha(ROOT/im['path'])==im['sha256'];images[im['path']]=im['sha256']
        for i,w in enumerate(r['witnesses']):
            assert (w['page']-1)*16+w['cell']==i
            assert w['source_path']==r['source_path'] and w['hash_verified'] is True
            assert any(all(f[k]==w[k] for k in ['frame_id','pts_s','rgb_sha256']) for f in traces[r['source_path']]['frames'])
    assert len(images)==59 and sum(len(r['witnesses']) for r in receipts)==674
    overlap=read(OUT/'overlap_adjudication.json')
    assert overlap['status']=='distinct_observable_scenes_single_review'
    registry=read(BASE/'content_release_v1/source_registry.json')['sources']
    split_registry=[dict(source_group_id=s['source_group_id'],split=s['split']) for s in registry if s['source_group_id']]
    groups={}
    for s in split_registry:groups.setdefault(s['source_group_id'],set()).add(s['split'])
    assert all(len(v)==1 for v in groups.values())
    checked_sources=set();ledger=[];admitted=[];requests=[];strata=[]
    for m in meta:
        q=dict(m['original']);a=m['source_identity'];source=q['source_path']
        if source not in checked_sources:
            asset=ROOT/a['media_path'];assert asset.stat().st_size==a['bytes'] and sha(asset)==a['sha256']
            checked_sources.add(source)
        q.update(source_group_id=a['source_group_id'],parent_identity_status=a['parent_identity_status'],
            content_domain=a['content_domain'],media_path=a['media_path'],media_sha256=a['sha256'])
        d=dm.get(q['id']);flags=list(m['previsual_flags']);witnesses=[]
        if 'weak_cross_split_visual_candidate_pending_adjudication' in flags:
            flags.remove('weak_cross_split_visual_candidate_pending_adjudication')
        if d:
            assert d['stratum'] in ['F','T','C','S'] and d['note'] and d['witness_cells']
            receipt=receipt_map[source]
            assert len(set(d['witness_cells']))==len(d['witness_cells'])
            for i in d['witness_cells']:
                w=dict(receipt['witnesses'][i]);assert w['pts_s']<=q['query_time_s']
                w['sheet']=receipt['images'][w['page']-1]
                witnesses.append(w)
            stratum=d['stratum'];note=d['note']
        else:
            stratum='S';note='未进入本轮图审：'+', '.join(flags)
        include=stratum!='S' and not flags
        cohort='instructional_supplement' if source.startswith(('COIN/','youcook2/','hirest/')) else 'natural_short_core'
        q['curation']=dict(stratum=stratum,note=note,witnesses=witnesses,previsual_flags=m['previsual_flags'],
            remaining_flags=flags,source_time_audit=m['source_time_audit'],cohort=cohort)
        q['admission']=dict(include_in_roster=include,independent_human_review=False,model_outputs_seen=False,
            stage='single_agent_model_canvas_qualification',formal_model_execution_frozen=False,
            exclusion_reason=None if include else note)
        assert official_identity(q)==official_identity(m['original'])
        ledger.append(q)
        if not include:continue
        assert a['parent_identity_status']=='verified' and groups[a['source_group_id']]=={'holdout'}
        q['review']=dict(verdict='supported_interpretation' if stratum=='C' else 'supported_observation',
            reviewer='Codex single-agent bounded native-canvas review; not independent human adjudication',
            note=note,flags_pending=[],model_outputs_seen=False,witnesses=witnesses,stratum=stratum,
            sampling_fps=1.,input_resolution_hw=[192,256],physical_causal_ground_truth_claimed=False)
        admitted.append(q)
        req=make_request(q,dict(source_path=source,media_path=a['media_path'],sha256=a['sha256']),traces[source])
        req['input_protocol']='holdout_1fps_192x256_data_roster_locked_execution_pending';requests.append(req)
        frames=req['model_visible']['frames'];pairs=[frames[i:i+2] for i in range(0,len(frames),2)]
        recent=[f for pair in pairs[-8:] for f in pair]
        intervals=q['evidence_intervals'] or (m['source_time_audit'] or {}).get('answer_intervals') or []
        fullhits=[any(a<=f['pts_s']<=b for f in frames) for a,b in intervals]
        recenthits=[any(a<=f['pts_s']<=b for f in recent) for a,b in intervals]
        if len(pairs)<=8:pressure='no_8pair_capacity_pressure'
        elif not intervals:pressure='author_interval_unavailable'
        elif not all(fullhits):pressure='full_sampling_misses_some_author_intervals'
        elif not any(recenthits):pressure='recent_misses_all_author_intervals'
        elif not all(recenthits):pressure='recent_misses_some_author_intervals'
        else:pressure='recent_covers_all_author_intervals'
        recent_ids={f['frame_id'] for f in recent}
        gap=max(0,q['query_time_s']-max(b for a,b in intervals)) if intervals else None
        strata.append(dict(id=q['id'],source_group_id=q['source_group_id'],cohort=cohort,stratum=stratum,
            official_task=q['task'],query_time_s=q['query_time_s'],real_frames=len(frames),pairs=len(pairs),
            author_intervals_used=intervals,author_interval_hits_full=fullhits,author_interval_hits_recent=recenthits,
            evidence_gap_proxy_s=gap,pressure_proxy=pressure,
            reviewer_witnesses_outside_recent=sum(w['frame_id'] not in recent_ids for w in witnesses),
            requires_old_evidence_proven=False))
    assert len(ledger)==94 and len(admitted)+sum(not q['admission']['include_in_roster'] for q in ledger)==94
    assert {q['id'] for q in admitted}=={q['id'] for q in requests}
    natural=[q for q in admitted if q['curation']['cohort']=='natural_short_core']
    instruction=[q for q in admitted if q['curation']['cohort']=='instructional_supplement']
    save(OUT/'all94_final_dispositions.json',ledger);save(OUT/'admitted.json',admitted)
    save(OUT/'natural_core.json',natural);save(OUT/'instructional_supplement.json',instruction)
    save(OUT/'model_requests.json',requests);save(OUT/'strata.json',strata)
    save(OUT/'scorer_only_key.json',[dict(id=q['id'],answer_index=q['answer_index'],stratum=q['curation']['stratum'],
        source_group_id=q['source_group_id'],cohort=q['curation']['cohort']) for q in admitted])
    protocol=dict(sampling_fps=1.,canvas_hw=[192,256],pairing_policy='two_real_or_snapshot_only_last_duplicate',
        query_cutoff_policy='pts_not_after_query',readout_layout='existing external-memory chronological labels and destination coordinate heuristic',
        generation=dict(max_new_tokens=8,do_sample=False,enable_thinking=False),visual_token_budget=384,
        full_reference_limits=dict(context_tokens=8192,visual_tokens=6144,nodes=128,suffix_tokens=128),
        input_trace_sha256=digest([r['model_visible'] for r in requests]),prompt_template_sha256=digest([prompt(q) for q in admitted]),
        model_revision=None,processor_revision=None,precision='float32',real_interface_receipt_sha256=None)
    bundle=dict(status='data_roster_locked_execution_admission_pending',schema='2b-memory-eval-v1',evaluation_scope='holdout',
        rows=admitted,original_rows=[official_identity(q) for q in original],split_registry=split_registry,
        original_rows_sha256=digest([official_identity(q) for q in original]),split_registry_sha256=digest(split_registry),input_protocol=protocol)
    save(OUT/'execution_bundle_draft.json',bundle)
    issues=freeze_issues(bundle)
    assert set(issues)=={'bundle_not_frozen','protocol_missing:model_revision','protocol_missing:processor_revision',
        'protocol_missing:real_interface_receipt_sha256','invalid_protocol_hash:real_interface_receipt_sha256'},issues
    rules=dict(status='locked_before_any_holdout_model_output',primary='Natural core F/T observations video-macro accuracy; C contextual interpretation separately.',
        instruction='All instructional examples separately reported; montage, subtitles and future reference anchors do not prove old-memory dependence.',
        aggregate='Report all51 only as labeled secondary aggregate, never mix instructional/simulated/future/identity-unresolved rows into natural causal accuracy.',
        scoring='Existing strict single A-E choice; format, missing, timeout and OOM count wrong. Keep all reference failures.',
        paired='Both correct, correction, new loss, both wrong; format-only changes separately shown.',
        uncertainty='Canonical-video cluster bootstrap, same resampled groups across arms, seed20261019,2000 replicates,95percent percentile intervals.',
        fixed_arms=['native_original','full_pairwise','M0','M1','M2','text_only','CM384','CMbytes322'],
        fixed_candidate=dict(q=64,m=8,gamma=.9,alpha=.8,CM384=384,CMbytes322=322,payload_ceiling_bytes=1600000),
        no_tuning_or_selection_using_holdout_answers=True,no_training=True,no_gpu_authorization_implied=True,
        long_reference_preflight='Native original helper previously has3072 context hard guard; adapt and preflight separately before allowing the long supplement. Never silently drop its old frames.',
        next='Model-specific CPU admission and fixed GPU budget first; W3 retrieval remains next mechanism slot, not started here.')
    save(OUT/'analysis_rules.json',rules)
    summary=dict(status='all94_accounted_data_roster_locked_no_inference',original_questions=94,original_videos=40,
        accepted=counts(admitted),natural_core=counts(natural),instructional_supplement=counts(instruction),
        primary_FT=counts([q for q in natural if q['curation']['stratum'] in ['F','T']]),
        supplementary_questions=94-len(admitted),visual_reviewed_questions=76,metadata_only_dispositions=18,
        previsual_flag_counts=dict(Counter(f for m in meta for f in m['previsual_flags'])),
        memory_pressure_proxy_counts=dict(Counter(s['pressure_proxy'] for s in strata)),
        reviewer_witnesses_outside_recent_questions=sum(s['reviewer_witnesses_outside_recent']>0 for s in strata),
        max_pairs=max(s['pairs'] for s in strata),max_visual_tokens=max(s['pairs']*48 for s in strata),
        actual_sheets=59,actual_selected_frames=674,images_disk_bytes=sum((ROOT/p).stat().st_size for p in images),
        models_loaded=0,gpu_jobs=0,cluster_operations=0,new_media_bytes=0,formal_execution_frozen=False,
        execution_admission_issues=issues,overlap_adjudication=overlap['status'],
        limitations=['Single assistant review; independent human adjudication absent.',
            'F/T/C denote evidence types, not statistical difficulty or intervention causality.',
            'Old witness absence from recent window is a proxy, not demonstrated answer dependence.',
            'Only3 qualified sources are longer than2minutes, all instructional; multi-minute natural causal challenge still absent.',
            'Known canonical IDs and bounded overlap review do not prove distinct filming sessions or absence of pretraining contamination.',
            'CPU working-set1GiB was a target, not measured process peak; no measured RAM-peak claim.'])
    save(OUT/'summary.json',summary)
    with (OUT/'全部94题资格账本.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','source_group_id','question','official_answer','query_time_s','stratum','cohort','admitted','note']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for q in ledger:w.writerow(dict(id=q['id'],source_group_id=q['source_group_id'],question=q['question'],
            official_answer=q['options'][q['answer_index']],query_time_s=q['query_time_s'],stratum=q['curation']['stratum'],
            cohort=q['curation']['cohort'],admitted=q['admission']['include_in_roster'],note=q['curation']['note']))
    names=['original94.json','previsual_metadata94.json','annotation_plan.json','visual_decisions.jsonl','overlap_adjudication.json',
        'sampling_1fps.json','image_receipts.json','all94_final_dispositions.json','admitted.json','natural_core.json',
        'instructional_supplement.json','model_requests.json','strata.json','scorer_only_key.json','execution_bundle_draft.json',
        'analysis_rules.json','summary.json','全部94题资格账本.csv']
    save(OUT/'DATA_ROSTER_LOCK.json',dict(status='single_review_data_locked_execution_pending',
        file_sha256={n:sha(OUT/n) for n in names},image_sha256=images,
        source_sha256={**plan['source_sha256'],str(__file__.replace(str(ROOT)+'\\','')):sha(ROOT/'experiments/2b-memory-eval/qualify_holdout.py')},
        source_registry_sha256=sha(BASE/'content_release_v1/source_registry.json'),
        model_outputs_seen=False,original235_unchanged=True,models_loaded=0))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
