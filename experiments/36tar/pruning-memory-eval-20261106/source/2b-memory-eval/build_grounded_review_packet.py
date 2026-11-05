"""Join recorded visual verdicts to immutable development questions and receipts.

This produces an annotation snapshot, never a scored/frozen evaluation bundle.
Author-guided witness frames are for annotation only, not model input selection.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from score_memory_eval import digest
from evaluation_gate import freeze_issues, official_identity


def main():
    root=Path(__file__).resolve().parents[2]
    base=root/'runs/development_audit_20261017_02'
    output=root/'runs/visual_review_20261017_01-sequential'
    candidate=root/'experiments/2b-memory-eval/data/candidate_manifest.json'
    all_rows=json.loads(candidate.read_text(encoding='utf-8'))
    rows=[r for r in all_rows if r['split']=='development']
    verdicts=json.loads((Path(__file__).parent/'data/development_visual_review_20261017.json').read_text(encoding='utf-8'))
    expected={r['id'] for r in rows if r['dataset']=='nextgqa'}
    if len(verdicts)!=len(expected) or {v['id'] for v in verdicts}!=expected:
        raise ValueError('every original GQA development ID must have exactly one verdict')
    by_id={v['id']:v for v in verdicts}
    previous=json.loads((base/'annotation_review_with_gqa.json').read_text(encoding='utf-8'))
    ovo={v['id']:v for v in previous['questions'] if v['id'].startswith('ovo:')}
    receipts=[]
    for folder in (output,root/'runs/visual_review_20261017_01-followup'):
        receipt=json.loads((folder/'review_receipts.json').read_text(encoding='utf-8'))
        for source in receipt['sources']:
            for frame in source['frames']:
                receipts.append(dict(source_path=source['source_path'],
                    image=str(folder/source['image']),**frame))
    result=[]
    for row in rows:
        item=dict(id=row['id'],original_row_sha256=digest(row),original=row,
                  second_reviewer_status='not_performed',
                  reviewer='Codex visual inspection; not an independent human annotation',
                  model_outputs_seen=False)
        if row['id'] in by_id:
            manual=by_id[row['id']]
            witnesses=[f for f in receipts if f['source_path']==row['source_path'] and
                any(a<=f['pts_s']<=b for a,b in manual['windows'])]
            if not witnesses or any(not f['hash_verified'] or f['pts_s']>row['query_time_s'] for f in witnesses):
                raise ValueError('visual review lacks actual prefix witnesses')
            item.update(manual,author_evidence_intervals=row['evidence_intervals'],
                verified_necessary_evidence_intervals=None,witnesses=witnesses,
                provisional_role='diagnostic_candidate' if manual['verdict'].startswith('supported_') else 'adjudication_only',
                scope='single_reviewer_sparse_actual_1fps_frames_not_all_video_frames')
        else:
            review=ovo[row['id']]
            item.update(previous_review=review,
                verdict='previous_review_carried_forward_no_new_images',
                provisional_role='diagnostic_candidate' if row['id'] in ('ovo:506','ovo:145') else 'adjudication_or_separate_control',
                scope='inherited evidence; this segment did not re-view OVO media')
        result.append(item)
    report=dict(status='development_annotation_snapshot_not_frozen_for_evaluation',
        candidate_manifest_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        original_questions_answers_splits_unchanged=True,
        annotation_only_author_guided_not_model_retrieval=True,
        pretrained_model_answers_seen=0,questions=result,
        gqa_verdict_counts=dict(Counter(v['verdict'] for v in verdicts)),
        gqa_main_review_sources=12,gqa_main_preview_frames=159,
        gqa_followup_sources=2,gqa_followup_preview_frames=19,
        formal_frozen_questions=0)
    (output/'development_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['id','dataset','source_path','question','official_answer','verdict','provisional_role','observed','limit','second_reviewer','original_row_sha256']
    with (output/'开发集46题审核结果.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for item in result:
            row=item['original'];old=item.get('previous_review',{})
            writer.writerow(dict(id=row['id'],dataset=row['dataset'],source_path=row['source_path'],
                question=row['question'],official_answer=row['options'][row['answer_index']],
                verdict=item['verdict'],provisional_role=item['provisional_role'],
                observed=item.get('observed',old.get('observed',old.get('visual_observation',old.get('note','')))),
                limit=item.get('limit',old.get('readiness','')),second_reviewer='未进行',
                original_row_sha256=item['original_row_sha256']))
    media={f['source_path']:f for name in ('media_inventory.json','gqa_media_inventory.json')
           for f in json.loads((base/name).read_text(encoding='utf-8'))['files']}
    traces={v['source_path']:v for v in json.loads((base/'sampling_all20_05fps.json').read_text(encoding='utf-8'))['videos']}
    proposed=[]
    for item in result:
        if item['provisional_role']!='diagnostic_candidate':
            continue
        row=dict(item['original'])
        if row['dataset']=='nextgqa':
            witnesses=item['witnesses']
            verdict=item['verdict'];note=item['observed']+' 限制：'+item['limit']
        else:
            old=item['previous_review']
            intervals=old.get('verified_necessary_evidence_intervals') or [old['provisional_inspection_window_s']]
            witnesses=[dict(f,source_path=row['source_path'],hash_verified=True)
                for f in traces[row['source_path']]['frames']
                if any(a<=f['pts_s']<=b for a,b in intervals) and f['pts_s']<=row['query_time_s']]
            verdict='supported_observation';note=old.get('observed',old.get('visual_observation'))
        row.update(source_group_id='candidate-file:'+row['dataset']+':'+row['source_id'],
            parent_identity_status='not_verified_as_independent_recording',
            media_sha256=media[row['source_path']]['sha256'],
            review=dict(verdict=verdict,reviewer=item['reviewer'],note=note,
                model_outputs_seen=False,flags_pending=[],witnesses=witnesses,
                second_review='not_performed',
                witness_scope='annotation evidence; not automatic retrieval or proof of final input coverage'))
        proposed.append(row)
    registry=list({(r['dataset'],r['source_id'],r['split']):dict(
        source_group_id='candidate-file:'+r['dataset']+':'+r['source_id'],split=r['split']) for r in all_rows}.values())
    originals=[official_identity(r) for r in rows]
    plan=dict(status='pre_freeze_development_plan',evaluation_scope='development_diagnostic',
        original_rows=originals,original_rows_sha256=digest(originals),
        split_registry=registry,split_registry_sha256=digest(registry),rows=proposed,
        input_protocol=dict(model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
            processor_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',precision='float32',
            sampling_fps=None,canvas_hw=[192,256],
            pairing_policy='two_real_or_snapshot_only_last_duplicate',
            query_cutoff_policy='pts_not_after_query',readout_layout=None,
            prompt_template_sha256=None,generation=None,visual_token_budget=None,
            input_trace_sha256=None,real_interface_receipt_sha256=None),
        limitations=['Provisional selection before model outputs; all 46 original items retained in audit.',
            'Source-file groups are not verified independent parent recordings.',
            '1FPS annotation witnesses do not certify final 0.5/1FPS input readability.',
            'Interpretation labels and small development sample do not establish physical causal inference.'])
    (output/'development_plan_not_frozen.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    readiness=dict(status='blocked_not_frozen',formal_frozen_questions=0,
        proposed_diagnostic_questions=len(proposed),blocking_issues=freeze_issues(plan),
        next_actions=[
            '恢复SSH后核原20分钟2B接口验收；无新增GPU授权前不扩大实验。',
            '用同一24题诊断清单检查已预定0.5/1FPS两设置的可读性和成本，选一次共同输入；不继续扫采样率。',
            '固定实际投影Token/位置/模板/生成及trace身份，按审核层次报告完整参考失败。',
            'W1再比较M0/M1/M2同预算记忆；原22道争议/特殊控制仍在46题原始审计中，交独立裁决而非自动重截图。'])
    (output/'freeze_readiness.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(questions=len(result),gqa_verdict_counts=report['gqa_verdict_counts'],
        diagnostic_candidates=sum(v['provisional_role']=='diagnostic_candidate' for v in result),
        formal_frozen_questions=0),ensure_ascii=False))


if __name__=='__main__':
    main()
