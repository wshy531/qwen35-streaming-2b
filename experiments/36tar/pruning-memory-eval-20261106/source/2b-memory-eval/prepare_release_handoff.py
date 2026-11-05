"""Prepare bounded review worklists and refreshed development admission plan.

No messages are sent and no missing review/interface receipt is fabricated.
Known canonical groups replace the previous file-only registry. Unknown Ego
parents get explicit asset-only development IDs and remain formal-scope blocked.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path

from evaluation_gate import freeze_issues,official_identity
from score_memory_eval import digest


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    release=run/'content_release_v1'
    annotated=json.loads((release/'question_annotation_overlay.json').read_text(encoding='utf-8'))['questions']
    by_id={q['id']:q for q in annotated}
    sources=json.loads((release/'source_registry.json').read_text(encoding='utf-8'))['sources']
    media={m['source_path']:m for m in json.loads((release/'media_inventory.json').read_text(encoding='utf-8'))['files']}
    old=root/'runs/visual_review_20261017_01-sequential/development_plan_not_frozen.json'
    plan=copy.deepcopy(json.loads(old.read_text(encoding='utf-8')))
    registry=[dict(source_group_id=s['source_group_id'] or 'unresolved-asset:'+s['dataset']+':'+s['source_path'],
        split=s['split'],parent_identity_status=s['parent_identity_status']) for s in sources]
    for row in plan['rows']:
        item=by_id[row['id']]
        row['source_group_id']=item['source_group_id'] or 'unresolved-asset:'+row['dataset']+':'+row['source_path']
        row['parent_identity_status']=item['parent_identity_status']
        row['media_sha256']=media[row['source_path']]['sha256']
        # The prior actual visual evidence is retained. This updates identity
        # only and does not pretend that final 2B readability was validated.
    plan.update(status='pre_freeze_development_plan',split_registry=registry,
        split_registry_sha256=digest(registry),
        identity_refresh_basis='content_release_v1/source_registry.json',
        draft_input_export_receipt_sha256=hashlib.sha256((run/'draft_input_export/receipt.json').read_bytes()).hexdigest(),
        no_final_input_protocol_or_true_interface_receipt=True)
    out=run/'handoff';out.mkdir(exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('development_plan_v2_not_frozen.json',plan)
    save('development_admission_remaining.json',dict(status='not_frozen',
        diagnostic_questions=len(plan['rows']),known_canonical_sources=len({q['source_group_id'] for q in plan['rows'] if q['parent_identity_status']=='verified'}),
        unresolved_development_asset_ids=[q['id'] for q in plan['rows'] if q['parent_identity_status']!='verified'],
        blocking_issues=freeze_issues(plan),formal_model_evaluation_allowed=False,
        note='No need to resolve all 10 Ego parents to run known-source GQA/COIN/CrossTask comparisons. Retain unresolved raw rows separately.'))
    fields=['id','split','source_path','query_time_s','question','options_json','official_answer',
        'current_verdict','flags','media_path','prior_witnesses_json','required_action']
    worklists={
        '留出94题_独立审核用.csv':[q for q in annotated if q['original']['split']=='holdout'],
        '比较疑点与完整题干_复核用.csv':[q for q in annotated if q['original']['split']=='comparison' and
            (q['provisional_role'] in ('adjudication_or_separate_control','partial_question_diagnostic_only') or
             q['id'] in ('ovo:624','ovo:609'))],
    }
    binding_path=run/'review_witness_input_binding.json'
    input_binding_ids=set()
    if binding_path.exists():
        technical=json.loads(binding_path.read_text(encoding='utf-8'))
        input_binding_ids=set(technical['summary']['comparison']['supported_with_no_actual_witness'])
        items=worklists['比较疑点与完整题干_复核用.csv']
        present={q['id'] for q in items}
        items.extend(q for q in annotated if q['id'] in input_binding_ids and q['id'] not in present)
    for name,items in worklists.items():
        with (out/name).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for q in items:
                r=q['original'];review=q['review'];is_negative=r['task']=='HLD'
                writer.writerow(dict(id=q['id'],split=r['split'],source_path=r['source_path'],query_time_s=r['query_time_s'],
                    question=r['question'],options_json=json.dumps(r['options'],ensure_ascii=False),
                    official_answer=r['options'][r['answer_index']],current_verdict=review['verdict'],
                    flags=' | '.join(q['flags']+(['no_exact_review_witness_in_draft_05fps']
                        if q['id'] in input_binding_ids else [])),media_path=media[r['source_path']]['media_path'],
                    prior_witnesses_json=json.dumps(review.get('witnesses',[]),ensure_ascii=False),
                    required_action='full_prefix_and_option_semantics_review_required_no_absence_inference_from_sparse_frames' if is_negative else
                        ('verify_same_fixed_input_readability_after_development_input_freeze_do_not_relabel_or_expand_current_screenshot_budget'
                         if q['id'] in input_binding_ids else 'review_answer_and_question_anchor_separately_record_actual_pts_or_keep_unresolved')))
    save('review_worklist_receipt.json',dict(status='worklists_prepared_not_review_completed_or_sent',
        counts={name:len(items) for name,items in worklists.items()},holdout_images_viewed=False,
        review_rule='same fixed observation/interpretation/unresolved/anticipation/negative rubric; no model outputs',
        stop_rule='independent pass then one explicit adjudication; unresolved remain outside primary qualified denominator',
        notes=['These reviewer/scorer files contain GT and must not be mounted as a model-visible prompt source.',
            'Keep all original failures and rejected questions with reasons; do not replace by answer success.',
            'Agent screenshot follow-up budget already used up for the present comparison pass.',
            'Annotation completion does not authorize GPU or alter the frozen input configuration.']))
    print(json.dumps(dict(worklists={name:len(items) for name,items in worklists.items()},
        refreshed_development_questions=len(plan['rows']),blocking_issues=freeze_issues(plan)),ensure_ascii=False))


if __name__=='__main__':main()
