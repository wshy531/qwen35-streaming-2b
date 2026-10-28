"""Freeze an outcome-blind 24-question API/cost panel and the full 478-question target."""
from collections import Counter
import hashlib
from audit_common import ROOT, OUT, read, save


def main():
    main_dir = ROOT/'runs/sember_main_prepare_20261025_01'
    questions = read(main_dir/'inputs/requests.json')
    videos = read(main_dir/'inputs/videos.json')
    expected = read(main_dir/'inputs/expected_prefixes.json')
    categories = sorted({q['question_category'] for q in questions})
    assert len(questions) == 478 and len(videos) == 128 and len(categories) == 8
    def length(q):
        return 'short' if q['question_time'] < 120 else 'medium' if q['question_time'] < 300 else 'long'
    # Fixed hash ordering of question identity. No labels, answers, or old correctness read.
    def key(q):
        return hashlib.sha256(('native-audit-v1:'+q['question_id']).encode()).hexdigest()
    selected = []
    for category in categories:
        for bucket in ('short','medium','long'):
            candidates = [q for q in questions if q['question_category'] == category and length(q) == bucket]
            assert candidates, (category,bucket)
            selected.append(min(candidates,key=key))
    assert len({q['question_id'] for q in selected}) == 24
    chosen_ids = {q['video_id'] for q in selected}
    panel_videos = [v for v in videos if v['video_id'] in chosen_ids]
    target = OUT/'inputs'
    target.mkdir(parents=True, exist_ok=True)
    save(target/'requests.json', selected)
    save(target/'videos.json', panel_videos)
    save(target/'expected_prefixes.json', {q['question_id']:expected[q['question_id']] for q in selected})
    save(target/'formal_requests.json', questions)
    save(target/'design.json', dict(primary_question='Does official one-shot reading outperform the modified full stream?',
        profile_questions=24, profile_sources=len(panel_videos), formal_target_questions=478,
        categories=dict(Counter(q['question_category'] for q in selected)),
        prefix_buckets=dict(Counter(length(q) for q in selected)),
        selected_without_labels_or_previous_answers=True, independent_test=False,
        conditions=['A_official540_default_video','B_official540_matched_pairs','B0_official530_matched_pairs'],
        existing_control='218526/R_all_history_stream on the identical selected questions',
        official_version_bridge_reason='5.3.0 raw multi-frame API fails; B0 versus B distinguishes the release change',
        dtype='FP32', do_sample=False, max_new_tokens=16, enable_thinking=False,
        attention='official sdpa, default eligible backends; log settings; old C forced math is disclosed',
        official_video_fps=2, official_video_max_frames=768,
        official_resize='untouched saved video_preprocessor_config size; no fixed canvas/letterbox',
        matched_fps=.5, matched_canvas=[288,384],
        no_future_frames=True, no_author_evidence_or_answers_in_runtime=True,
        no_training_or_prompt_search=True, no_context_truncation=True,
        no_overall_claim_from_profile=True, profile_only_before_full478_budget=True,
        policy='one fixed profile, preserve all failures; no automatic rerun or candidate sweep',
        max_formal_answers=72, max_warmup_answers=3, max_answers=75,
        max_generated_tokens=1200, pretrained_loads=3,
        GPU_authorized=False))
    print(dict(status='outcome_blind_panel_frozen',questions=24,sources=len(panel_videos),
        query_time_range=[min(q['question_time'] for q in selected),max(q['question_time'] for q in selected)]))


if __name__ == '__main__':
    main()
