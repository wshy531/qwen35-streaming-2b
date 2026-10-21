"""Predeclare comparisons before content qualification and model predictions."""
from prepare import OUT, ROOT, read, save, sha


def main():
    previous = ROOT/'runs/2b_short_event_prepare_20261020_01'
    source_lock = read(previous/'source_lock.json')
    dependency_lock = read(previous/'dependency_lock.json')
    for folder, lock in [(ROOT/'experiments/2b-short-event-v1',source_lock),
                         (ROOT/'experiments/2b-natural-eval-v1',dependency_lock)]:
        for name, digest in lock.items():
            assert sha(folder/name) == digest, name
    protocol = dict(
        status='prospective_analysis_only_NOT_GPU_authorization',
        candidate_roster_sha256=sha(OUT/'candidate_questions119.json'),
        admission='all pixel-supported original questions; no filtering by model correctness',
        raw_denominator=119, raw_sources=40,
        unsupported_and_ambiguous='retain original IDs, official labels, reasons and counts; do not silently delete',
        original_full_reference_failures='include if independently content-supported',
        sampling=dict(fps=1,canvas_hw=[192,256],letterbox=True,causal_prefix=True,
                      same_PTS_and_RGB_across_arms=True,tail_duplicate_explicit=True),
        models='same pinned Qwen3.5-2B FP32 weights, original and modified code in isolated processes',
        proposed_arms=dict(native=['native_original'],memory=[
            'full_pairwise_reference','M0','M1','M2','B02_cap4','reservoir4','text_only']),
        maximum_questions_per_arm=119,maximum_answers=952,maximum_generated_tokens=7616,
        pretrained_loads_proposal=2,
        candidate_visual_token_cap=384,full_reference_exempt=True,
        full_reference_storage='additional cache; cannot be counted as bounded streaming memory',
        fixed_parameters=dict(forest_threshold=.047536224126815796,
                              B02_open_pair_cap=4,reservoir_seed=20261020),
        frozen_candidate_source_lock_sha256=sha(previous/'source_lock.json'),
        frozen_dependency_lock_sha256=sha(previous/'dependency_lock.json'),
        primary='content-choice correct on all admitted F/T questions, source-group paired counts and intervals',
        primary_comparisons=['M2 vs M0','B02_cap4 vs M2','reservoir4 vs M0'],
        secondary=['C questions','strict exact-choice accuracy','format compliance',
                   'recent-role regressions','early-role corrections and losses',
                   'native_original vs full_pairwise_reference generated IDs and actual pixel inputs',
                   'all-source totals plus180s single-source supplement separately'],
        scorer='reuse frozen2b-short-event-v1/scoring.py measure; no free-form semantic rescoring',
        errors='count readout errors or unparseable outputs as failures and report separately',
        paired_units='conservative source/filming group after content-only overlap review',
        uncertainty=dict(seed=20261020,bootstrap_replicates=2000,
                         resampling_unit='source groups, retaining all questions and paired arms',
                         intervals='descriptive95percent percentile; no sample-size/power or multiple-test significance claim'),
        required_resource_fields=['model load seconds','per-question elapsed seconds','visual read tokens',
                                  'persistent candidate tensor bytes','peak allocated GPU bytes',
                                  'process RSS','forced seals/within/cross merges/root depth'],
        cost_limits='not a warmed interleaved speed benchmark; no end-to-end speed claim',
        necessity='author intervals and recent-only failures do not establish necessary old memory',
        long_video_gap='one180s source is not a broad multi-minute benchmark; report remaining gap',
        full_input_preflight_issue='inherited native_inputs has3072-token engineering guard;180s1FPS may exceed it. '
            'Do not truncate, lower only one arm FPS, or silently omit. Before GPU, explicitly preflight a '
            'new common full-input ceiling in a separate adapter or retain capacity failure in all-source ledger.',
        next_gate='finish blind pixel admission, actual prefix/capacity CPU preflight and scoped GPU budget approval',
        stop='one fixed T comparison, no threshold/cap/seed/prompt search on these results',
        GPU_jobs_submitted=0,model_answers_opened=0)
    save(OUT/'analysis_protocol.json',protocol)
    print('Analysis fixed before content verdicts or model answers; maximum952 answers, GPU not authorized.')


if __name__ == '__main__':
    main()
