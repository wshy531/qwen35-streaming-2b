"""Fixed-policy CPU coverage diagnostics, never prediction accuracy.

Use development/comparison metadata only. This audit cannot determine whether
an annotated span is necessary, a kept frame is readable, or the model uses it.
Policies and budgets are inherited; no thresholds/policies are tuned here.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path

from audit_sampling import query_snapshot
from audit_development_evidence import no_evidence_option_rule


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    release=run/'content_release_v1'
    questions=json.loads((release/'question_annotation_overlay.json').read_text(encoding='utf-8'))['questions']
    traces={t['source_path']:t for t in json.loads((release/'actual_sampling_traces.json').read_text(encoding='utf-8'))['videos']}
    result=[];negative=[]
    for item in questions:
        row=item['original']
        if row['split']=='holdout':continue
        if row['task']=='HLD':
            predicted=no_evidence_option_rule(row)
            negative.append(dict(id=row['id'],split=row['split'],rule_option_index=predicted,
                matched_original_label=predicted==row['answer_index'],video_was_not_used=True))
        if row['dataset']!='nextgqa':continue
        snap=query_snapshot(traces[row['source_path']]['frames'],row['query_time_s'],row['evidence_intervals'])
        prefix=[f for f in traces[row['source_path']]['frames'] if f['pts_s']<=row['query_time_s']]
        def any_span(policy=None):
            if policy is None:kept=prefix
            else:kept=[prefix[min(i,len(prefix)-1)] for p in snap[policy]['pair_indices'] for i in (2*p,2*p+1)]
            return any(a<=f['pts_s']<=b for a,b in row['evidence_intervals'] for f in kept)
        a,b=snap['M0']['all_verified_intervals_hit'],snap['M1']['all_verified_intervals_hit']
        result.append(dict(id=row['id'],split=row['split'],task=row['task'],
            provisional_role=item['provisional_role'],answer_support=item['review']['verdict'],
            author_intervals=row['evidence_intervals'],author_evidence_gap_s=row['evidence_gap_s'],
            required_intervals_independently_verified=False,
            full_all_spans_hit=snap['full_history_all_verified_intervals_hit'],
            M0_all_spans_hit=a,M1_all_spans_hit=b,
            full_any_span_hit=any_span(),M0_any_span_hit=any_span('M0'),M1_any_span_hit=any_span('M1'),
            M1_only_all_spans_hit=b and not a,M0_only_all_spans_hit=a and not b,
            full_real_frames=snap['sampled_real_frames'],pairs=snap['pairs'],
            actual_interval_sample_hits=snap['verified_interval_sample_hits'],
            M0_pair_indices=snap['M0']['pair_indices'],M1_pair_indices=snap['M1']['pair_indices']))
    summaries={}
    for split in ('development','comparison'):
        group=[r for r in result if r['split']==split]
        summaries[split]=dict(questions=len(group),tasks=dict(Counter(r['task'] for r in group)),
            full_all_spans_hit=sum(r['full_all_spans_hit'] for r in group),
            M0_all_spans_hit=sum(r['M0_all_spans_hit'] for r in group),
            M1_all_spans_hit=sum(r['M1_all_spans_hit'] for r in group),
            full_any_span_hit=sum(r['full_any_span_hit'] for r in group),
            M0_any_span_hit=sum(r['M0_any_span_hit'] for r in group),
            M1_any_span_hit=sum(r['M1_any_span_hit'] for r in group),
            M1_only_any_span_hit=sum(r['M1_any_span_hit'] and not r['M0_any_span_hit'] for r in group),
            M0_only_any_span_hit=sum(r['M0_any_span_hit'] and not r['M1_any_span_hit'] for r in group),
            M1_only_all_spans_hit=sum(r['M1_only_all_spans_hit'] for r in group),
            M0_only_all_spans_hit=sum(r['M0_only_all_spans_hit'] for r in group),
            author_gap_at_least_32s=sum(r['author_evidence_gap_s']>=32 for r in group),
            author_gap_at_least_60s=sum(r['author_evidence_gap_s']>=60 for r in group))
        summaries[split]['coverage_by_annotation_role']={role:dict(
            questions=len(subset),M1_only=sum(r['M1_only_all_spans_hit'] for r in subset),
            M0_only=sum(r['M0_only_all_spans_hit'] for r in subset))
            for role in sorted({r['provisional_role'] for r in group})
            if (subset:=[r for r in group if r['provisional_role']==role])}
    output=dict(status='actual_sample_and_fixed_policy_coverage_not_accuracy_or_capability_selection',
        policies=dict(M0='8 complete pairs FIFO',M1='4 recent plus 4 hierarchical long complete representatives'),
        sampling='existing draft .5FPS, same actual trace for both; not a finalized 2B input protocol',
        holdout_semantic_analysis_performed=False,summaries=summaries,questions=result,
        unable_option_shortcut=dict(rule='unique literal Unable to answer without video',
            questions=len(negative),matched_original_labels=sum(r['matched_original_label'] for r in negative),
            rows=negative,meaning='format-only diagnostic, not model answers or grounded refusal'),
        limits=['Author related spans are not proven jointly necessary.',
            'Coverage differences indicate potential probes, not improvement in answer quality.',
            'Any-span and all-span hits are both diagnostics; neither proves sufficient or necessary evidence.',
            'An M0 miss must remain in full-reference evaluation; do not select on model success.',
            'Evidence intervals are used after memory selection for auditing only, never for selecting memory.'])
    (run/'fixed_policy_actual_coverage.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(summaries=summaries,unable_option_shortcut={k:v for k,v in output['unable_option_shortcut'].items() if k!='rows'}),ensure_ascii=False))


if __name__=='__main__':main()
