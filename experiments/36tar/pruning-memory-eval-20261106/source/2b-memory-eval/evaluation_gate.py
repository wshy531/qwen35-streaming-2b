"""Structural evaluation admission: explicit identities, review and causal inputs.

Validates recorded evidence; it cannot certify a human/AI review was truthful.
Unreviewed official questions remain in the audit, not silently relabelled.
"""
import math
import re

from score_memory_eval import digest


HASH=re.compile(r'^[0-9a-f]{64}$')
OFFICIAL_FIELDS=('id','dataset','source_id','source_path','task','question','options',
                 'answer_index','query_time_s','split')


def official_identity(row):
    return {k:row.get(k) for k in OFFICIAL_FIELDS}


def freeze_issues(bundle):
    issues=[]
    if bundle.get('status')!='frozen_for_evaluation':
        issues.append('bundle_not_frozen')
    if bundle.get('evaluation_scope') not in ('development_diagnostic','comparison','holdout'):
        issues.append('evaluation_scope_missing')
    protocol=bundle.get('input_protocol',{})
    required=('model_revision','processor_revision','precision','sampling_fps',
              'canvas_hw','pairing_policy','query_cutoff_policy','readout_layout',
              'prompt_template_sha256','generation','visual_token_budget',
              'input_trace_sha256','real_interface_receipt_sha256')
    for field in required:
        if protocol.get(field) in (None,'',[],{}):
            issues.append('protocol_missing:'+field)
    for field in ('input_trace_sha256','real_interface_receipt_sha256','prompt_template_sha256'):
        if not HASH.fullmatch(str(protocol.get(field,''))):
            issues.append('invalid_protocol_hash:'+field)
    fps=protocol.get('sampling_fps')
    if isinstance(fps,bool) or not isinstance(fps,(float,int)) or not math.isfinite(fps) or fps<=0:
        issues.append('invalid_sampling_fps')
    if protocol.get('query_cutoff_policy')!='pts_not_after_query':
        issues.append('query_cutoff_not_causal')
    generation=protocol.get('generation',{})
    if not isinstance(generation,dict) or any(k not in generation for k in
        ('max_new_tokens','do_sample','enable_thinking')):
        issues.append('generation_not_fixed')
    else:
        maximum=generation['max_new_tokens']
        if isinstance(maximum,bool) or not isinstance(maximum,int) or maximum<=0:
            issues.append('invalid_generation_budget')
    rows=bundle.get('rows',[])
    if not rows or len({r.get('id') for r in rows})!=len(rows):
        issues.append('nonempty_unique_rows_required')
    originals=bundle.get('original_rows',[])
    original_map={r.get('id'):r for r in originals}
    if len(original_map)!=len(originals) or not originals:
        issues.append('original_question_snapshot_missing_or_duplicate')
    if bundle.get('original_rows_sha256')!=digest(originals):
        issues.append('original_question_snapshot_hash_mismatch')
    registry=bundle.get('split_registry',[])
    groups={}
    for entry in registry:
        groups.setdefault(entry.get('source_group_id'),set()).add(entry.get('split'))
    if not groups or bundle.get('split_registry_sha256')!=digest(registry):
        issues.append('split_registry_missing_or_hash_mismatch')
    if any(len(splits)>1 for splits in groups.values()):
        issues.append('source_recording_leaks_across_splits')
    expected_split='development' if bundle.get('evaluation_scope')=='development_diagnostic' else bundle.get('evaluation_scope')
    for row in rows:
        key=str(row.get('id'));prefix=key+':'
        original=original_map.get(row.get('id'))
        if original is None or official_identity(row)!=official_identity(original):
            issues.append(prefix+'original_question_or_label_changed')
        if not row.get('source_group_id'):
            issues.append(prefix+'source_group_missing')
        if row.get('split')!=expected_split or groups.get(row.get('source_group_id'))!={row.get('split')}:
            issues.append(prefix+'source_split_not_registered_or_wrong_scope')
        if bundle.get('evaluation_scope') in ('comparison','holdout') and row.get('parent_identity_status')!='verified':
            issues.append(prefix+'parent_recording_not_verified')
        if not HASH.fullmatch(str(row.get('media_sha256',''))):
            issues.append(prefix+'media_hash_missing')
        review=row.get('review',{})
        if review.get('verdict') not in ('supported_observation','supported_interpretation','verified_negative'):
            issues.append(prefix+'answer_evidence_unresolved')
        if not review.get('reviewer') or not review.get('note'):
            issues.append(prefix+'review_record_missing')
        if review.get('model_outputs_seen') is not False:
            issues.append(prefix+'selection_not_independent_of_model_outputs')
        if review.get('flags_pending'):
            issues.append(prefix+'adjudication_pending')
        if row.get('task')=='HLD' or review.get('verdict')=='verified_negative':
            if not review.get('full_prefix_reviewed') or not HASH.fullmatch(str(review.get('full_prefix_review_receipt_sha256',''))):
                issues.append(prefix+'negative_requires_full_prefix_review')
        else:
            witnesses=review.get('witnesses',[])
            if not witnesses:
                issues.append(prefix+'actual_visual_witness_missing')
            for witness in witnesses:
                pts=witness.get('pts_s');query=row.get('query_time_s')
                if (isinstance(pts,bool) or not isinstance(pts,(float,int)) or not math.isfinite(pts)
                    or isinstance(query,bool) or not isinstance(query,(float,int))
                    or not math.isfinite(query) or pts>query or pts<0):
                    issues.append(prefix+'witness_after_query_or_invalid')
                if witness.get('source_path')!=row.get('source_path') or witness.get('hash_verified') is not True or not HASH.fullmatch(str(witness.get('rgb_sha256',''))):
                    issues.append(prefix+'witness_identity_not_verified')
    return list(dict.fromkeys(issues))


def require_frozen(bundle):
    issues=freeze_issues(bundle)
    if issues:
        raise ValueError('evaluation admission failed: '+'; '.join(issues[:12]))


def require_prediction_identity(bundle,prediction):
    protocol=bundle['input_protocol']
    if prediction.get('evaluation_sha256')!=digest(bundle):
        raise ValueError('evaluation bundle identity mismatch')
    for field in ('input_trace_sha256','model_revision'):
        if prediction.get(field)!=protocol[field]:
            raise ValueError('prediction does not match frozen protocol: '+field)
