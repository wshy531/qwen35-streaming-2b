"""Join immutable raw candidates, media receipts, source identity and reviews.

This is a versioned local DATA content snapshot, not an evaluation freeze.
No model is loaded and no holdout images/semantic annotations are reviewed.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from score_memory_eval import digest

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/dataset_build_20261018_01'
EXPECTED_CANDIDATE_SHA='cd1d888eb0a0c00f300649f41800e60fd02b002e91f41f154b42e041fcc28161'

# Answer visibility is weaker than validation of a whole question's anchor,
# identity or consequence. These flags preserve rather than silently fix it.
PARTIAL_QUESTION={
    'nextgqa:val:6618792571:2':'release_of_handle_anchor_not_directly_seen',
    'nextgqa:val:5702201498:0':'sleep_state_anchor_not_independently_verified',
    'nextgqa:val:3066064005:1':'purple_clothing_identity_ambiguous',
    'nextgqa:val:3489439407:5':'second_person_identity_scope_uncertain',
    'nextgqa:val:4984417707:2':'kick_visible_goal_trajectory_not_fully_traced',
    'nextgqa:val:6163877860:4':'kick_off_sofa_anchor_not_directly_seen',
    'nextgqa:val:6329077812:3':'milk_residue_visible_spit_event_not_directly_seen',
    'nextgqa:val:9735768590:4':'smile_visible_tongue_trigger_not_directly_seen',
    'nextgqa:val:4132599924:1':'audio_change_not_checked',
    'nextgqa:val:5429688959:5':'singing_audio_not_checked',
    'nextgqa:val:5429688959:4':'singing_audio_not_checked',
    'ovo:72':'pink_cloth_visible_wiping_anchor_not_seen',
    'ovo:624':'pickup_not_directly_seen',
    'ovo:560':'all_five_ingredients_not_visually_identified',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
    if isinstance(value,list):return [portable(v) for v in value]
    if isinstance(value,str) and value.lower().startswith(str(ROOT).lower()+'\\'):
        return Path(value).relative_to(ROOT).as_posix()
    return value


def load_media_traces():
    base=ROOT/'runs/development_audit_20261017_02'
    media=[f for name in ('media_inventory.json','gqa_media_inventory.json')
        for f in read(base/name)['files']]
    media.extend(f for name in ('nextgqa-eval48-hfverified-20261018','ovo-eval32-20261018')
        for f in read(ROOT/'datasets/2b-memory-eval'/name/'download_manifest.json')['files'])
    traces=[]
    for v in read(base/'sampling_all20_05fps.json')['videos']:
        traces.append(dict(source_path=v['source_path'],frames=v['frames'],
            pts_origin_s=v['pts_origin_s'],fps=.5,canvas_hw=v['canvas_hw'],
            audit_source='runs/development_audit_20261017_02/sampling_all20_05fps.json'))
    for name in ('gqa48_actual_index.json','ovo32_actual_index.json'):
        for v in read(RUN/name)['videos']:
            traces.append(dict(source_path=v['source_path'],frames=v['frames_05fps'],
                pts_origin_s=v['first_pts_s'],fps=.5,canvas_hw=[192,256],
                audit_source='runs/dataset_build_20261018_01/'+name))
    if len(media)!=100 or len({m['source_path'] for m in media})!=100:
        raise ValueError('exactly 100 unique media receipts required')
    if len(traces)!=100 or len({t['source_path'] for t in traces})!=100:
        raise ValueError('exactly 100 unique decoded traces required')
    return {m['source_path']:m for m in media},{t['source_path']:t for t in traces}


def comparison_reviews(rows):
    manual=[]
    for name in ('comparison_visual_review_20261018.jsonl','ovo_comparison_visual_review_20261018.jsonl'):
        manual.extend(json.loads(line) for line in (ROOT/'experiments/2b-memory-eval/data'/name)
            .read_text(encoding='utf-8').splitlines() if line.strip())
    if len(manual)!=95 or {m['id'] for m in manual}!={r['id'] for r in rows if r['split']=='comparison'}:
        raise ValueError('comparison verdicts must cover all 95 original questions exactly once')
    by_id={m['id']:m for m in manual}
    for followup in read(RUN/'gqa_resolution_followup/visual_verdicts.json'):
        by_id[followup['id']]=dict(followup,initial_review=by_id[followup['id']],
            followup_budget_status='two_source_followup_budget_exhausted')
    receipts=defaultdict(list)
    for folder in ('gqa_comparison_sheets','ovo_comparison_sheets','gqa_resolution_followup'):
        for source in read(RUN/folder/'review_receipts.json')['sources']:
            for frame in source['frames']:
                receipts[source['source_path']].append(dict(frame,image=str(RUN/folder/source['image'])))
    result={}
    for row in rows:
        if row['split']!='comparison':continue
        m=by_id[row['id']];witnesses=[]
        for pts in m['witness_pts_s']:
            candidates=[f for f in receipts[row['source_path']] if abs(f['pts_s']-pts)<.0006]
            if not candidates:raise ValueError(f'{row["id"]}: witness not actually previewed: {pts}')
            f=candidates[-1]
            if not f.get('hash_verified',True) or f['pts_s']>row['query_time_s']+1e-9:
                raise ValueError('unverified or future witness')
            witnesses.append(dict(f,source_path=row['source_path'],hash_verified=True))
        result[row['id']]=dict(m,witnesses=witnesses,
            reviewer='Codex sparse-frame visual review; no independent human review',
            model_outputs_seen=False,second_reviewer_status='not_performed',
            verified_necessary_evidence_intervals=None,
            scope='annotation_only_not_model_input_selection_or_low_resolution_readability')
    return result


def main():
    candidate=ROOT/'experiments/2b-memory-eval/data/candidate_manifest.json'
    if sha(candidate)!=EXPECTED_CANDIDATE_SHA:raise ValueError('original candidate manifest changed')
    rows=read(candidate);media,traces=load_media_traces()
    registry=read(RUN/'source_registry.json')
    identities={v['source_path']:v for v in registry['sources']}
    if registry['cross_split_known_group_conflicts']:raise ValueError('known source leakage')
    development={q['id']:q for q in read(ROOT/'runs/visual_review_20261017_01-sequential/development_review.json')['questions']}
    comparisons=comparison_reviews(rows)
    temporal={q['id']:q for q in read(RUN/'ovo_comparison_temporal_final.json')['questions']}
    inventory=[];actual=[]
    for source,m in media.items():
        identity=identities[source];path=Path(m['local_path']).resolve()
        if not path.is_relative_to((ROOT/'datasets/2b-memory-eval').resolve()) or not path.is_file():
            raise ValueError('media missing or outside explicit data directory')
        if path.stat().st_size!=m['bytes']:raise ValueError('media size changed')
        qs=[r for r in rows if r['source_path']==source];trace=traces[source]
        frames=trace['frames']
        if not frames or any(f['pts_s']>max(q['query_time_s'] for q in qs)+1e-9 for f in frames):
            raise ValueError('empty or future sampled trace')
        if any(a['pts_s']>=b['pts_s'] or a['frame_id']>=b['frame_id'] for a,b in zip(frames,frames[1:])):
            raise ValueError('sampled trace not strictly increasing')
        inventory.append(dict(dataset=identity['dataset'],source_path=source,
            media_path=path.relative_to(ROOT).as_posix(),bytes=m['bytes'],sha256=m['sha256'],
            split=identity['split'],source_group_id=identity['source_group_id'],
            parent_identity_status=identity['parent_identity_status'],content_domain=identity['content_domain'],
            integrity_basis=m.get('integrity','recorded_local_sha; GQA additionally pinned mirror SHA and author ZIP CRC'),
            receipt=portable(m)))
        actual.append(dict(trace,source_sha256=m['sha256'],split=identity['split'],
            trace_sha256=digest(trace),protocol='draft_.5fps_canvas_192x256_not_model_frozen'))
    annotated=[]
    for row in rows:
        identity=identities[row['source_path']];flags=[]
        if row['split']=='holdout':
            review=dict(verdict='not_visually_reviewed_holdout',model_outputs_seen=False,
                second_reviewer_status='not_performed');role='holdout_pending_independent_review'
        elif row['split']=='development':
            review=dict(development[row['id']]);review.pop('original',None)
            role='inherited_development_diagnostic' if review['provisional_role']=='diagnostic_candidate' else 'adjudication_or_separate_control'
        else:
            review=comparisons[row['id']]
            if review['verdict'] in ('unresolved','negative_unverified'):role='adjudication_or_separate_control'
            elif review['verdict']=='future_action_task':role='anticipation_separate_from_recall'
            elif identity['content_domain']=='naturalistic_recorded_video':role='provisional_naturalistic_diagnostic'
            else:role='supplementary_control'
        if row['id'] in PARTIAL_QUESTION:flags.append(PARTIAL_QUESTION[row['id']])
        if identity['parent_identity_status']!='verified':flags.append('parent_recording_unresolved_blocks_formal_source_core')
        if row['split']!='holdout':
            if row['id'] in ('ovo:486','ovo:515','ovo:496'):flags.append('burned_in_text_present')
            if row['id']=='ovo:301':flags.append('referent_may_be_visible_positive_options_do_not_cover_countertop')
            if row['id']=='ovo:471':flags.append('literal_E_prefix_inside_original_option')
            if row['task']=='HLD':flags.extend(['full_prefix_negative_review_missing','unable_option_shortcut_diagnostic_required'])
            if review['verdict']=='supported_interpretation':flags.append('contextual_inference_not_direct_physical_causal_ground_truth')
        if row['id'] in PARTIAL_QUESTION and role=='provisional_naturalistic_diagnostic':role='partial_question_diagnostic_only'
        annotated.append(dict(id=row['id'],original_row_sha256=digest(row),original=row,
            source_group_id=identity['source_group_id'],parent_identity_status=identity['parent_identity_status'],
            content_domain=identity['content_domain'],review=review,flags=flags,provisional_role=role,
            source_timing_audit=temporal.get(row['id']),
            formal_evaluation_eligible=False))
    out=RUN/'content_release_v1';out.mkdir(exist_ok=False)
    def write(name,value):
        (out/name).write_text(json.dumps(portable(value),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write('media_inventory.json',dict(status='local_receipts_and_current_sizes_verified',files=inventory))
    write('actual_sampling_traces.json',dict(status='actual_cpu_.5fps_traces_not_model_inputs_frozen',videos=actual))
    write('source_registry.json',registry)
    write('question_annotation_overlay.json',dict(status='fixed_content_snapshot_not_frozen_for_model_evaluation',
        candidate_manifest_sha256=EXPECTED_CANDIDATE_SHA,questions=annotated))
    splitstats={}
    for split in ('development','comparison','holdout'):
        qs=[q for q in annotated if q['original']['split']==split];vs=[v for v in inventory if v['split']==split]
        splitstats[split]=dict(media_files=len(vs),bytes=sum(v['bytes'] for v in vs),questions=len(qs),
            annotation_records_completed=sum(q['review']['verdict']!='not_visually_reviewed_holdout' for q in qs),
            known_canonical_source_ids=sum(v['parent_identity_status']=='verified' for v in vs),
            verdict_counts=dict(Counter(q['review']['verdict'] for q in qs)),
            role_counts=dict(Counter(q['provisional_role'] for q in qs)))
    summary=dict(status='fixed_local_dataset_content_v1_not_model_evaluation_freeze',
        version='20261018-local-v1',candidate_manifest_sha256=EXPECTED_CANDIDATE_SHA,
        total_media_files=len(inventory),total_media_bytes=sum(v['bytes'] for v in inventory),
        total_questions=len(annotated),actual_05fps_frames=sum(len(t['frames']) for t in actual),
        known_canonical_source_ids=registry['known_identifier_sources'],unresolved_parent_sources=registry['unresolved_sources'],
        known_source_cross_split_conflicts=registry['cross_split_known_group_conflicts'],splits=splitstats,
        comparison_verdict_counts=dict(Counter(q['review']['verdict'] for q in annotated if q['original']['split']=='comparison')),
        comparison_partial_question_flags=[q['id'] for q in annotated if q['original']['split']=='comparison' and q['id'] in PARTIAL_QUESTION],
        media_on_cluster=False,pretrained_2b_loads=0,pretrained_2b_answers=0,formal_frozen_questions=0,
        remaining=['10 Ego parent IDs unresolved; do not assume 100 independent recordings.',
            '94 holdout questions lack visual adjudication; no holdout contact sheets used.',
            'Single-agent sparse review, no independent human check; uncertain items retained.',
            'No true 2B interface receipt or final sampling/token/layout/generation protocol.',
            'Natural why/intent labels are not interventional physical causality ground truth.',
            'Raw annotation and media provenance remain subject to original upstream licensing.'])
    write('summary.json',summary)
    fields=['id','split','dataset','source_path','question','official_answer','verdict','provisional_role','flags','observed','limit','source_group_id']
    with (out/'全部235题审核表.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for q in annotated:
            r=q['original'];review=q['review'];old=review.get('previous_review',{})
            writer.writerow(dict(id=q['id'],split=r['split'],dataset=r['dataset'],source_path=r['source_path'],
                question=r['question'],official_answer=r['options'][r['answer_index']],verdict=review['verdict'],
                provisional_role=q['provisional_role'],flags=' | '.join(q['flags']),
                observed=review.get('observed',old.get('visual_observation',old.get('observed',''))),
                limit=review.get('limit',old.get('readiness','')),source_group_id=q['source_group_id']))
    write('content_hashes.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
