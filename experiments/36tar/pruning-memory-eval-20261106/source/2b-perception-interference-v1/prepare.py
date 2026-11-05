"""Freeze three pre-existing development anchors without consulting new model outputs."""
from common import ROOT, PREP, INPUTS, OLD, ARMS, read, save, sha


CURRENT = [
    ('In the last committed frame, what surface is the outstretched finger pointing at?',
     ['The washing machine touchscreen.', 'A detergent bottle.', 'A shopping cart handle.',
      'A clothes hanger.', 'The floor.']),
    ('Which two colors dominate the displayed kitchen tools in these recent frames?',
     ['White and green.', 'Red and black.', 'Blue and gray.', 'Purple and yellow.', 'Pink and brown.']),
    ('What items occupy the foreground rolling display in the last committed frame?',
     ['Suitcases and travel bags.', 'Pajamas.', 'Bath towels.', 'Cooking pots.', 'Plates and cutlery.']),
]


def main():
    if INPUTS.exists():
        raise FileExistsError('retain previous inputs; do not silently regenerate')
    drafts = read(OLD / 'paired_current_drafts_LOCAL_ONLY.json')
    index = {r['query']['id']: r for r in read(OLD / 'case_review_index_LOCAL_ONLY.json')}
    selection = ROOT / 'runs/sember_prepare_20261025_01/selection'
    labels = {r['question_id']: r for r in read(selection / 'labels_LOCAL_ONLY.json')}
    videos = {r['video_id']: r for r in read(selection / 'videos.json')}
    assert len(drafts) == len(CURRENT) == 3
    pairs, tasks, schedule, review = [], [], [], []
    cards = sorted((OLD / 'review_cards_LOCAL_ONLY').glob('*.png'))
    for i, (draft, (question, options)) in enumerate(zip(drafts, CURRENT)):
        row = index[draft['old_question_id']]
        old = row['query']
        card = row['current_review_card']
        image = next(p for p in cards if sha(p) == draft['review_image_sha256'])
        qid = old['id'][7:]
        video_id = old['source_path'].split('/')[-1][:-4]
        donor = drafts[(i + 1) % len(drafts)]
        pair_id = f'P{i + 1:02d}'
        recent = card['recent4_frames']
        assert len(recent) == 8 and recent[-1]['pts_s'] == draft['current_reference_time_s']
        assert all(f['pts_s'] <= old['query_time_s'] for f in recent)
        assert videos[video_id]['sha256']
        pairs.append(dict(pair_id=pair_id, old_question_id=old['id'],
            source_path=old['source_path'], media_sha256=videos[video_id]['sha256'],
            media_bytes=videos[video_id]['bytes'], query_time_s=old['query_time_s'],
            committed_through_s=card['actual_committed_prefix']['committed_through_s'],
            committed_pairs=card['actual_committed_prefix']['committed_pairs'],
            pending_frame=card['actual_committed_prefix']['pending_frame'],
            recent_frames=recent, image_relative_path=image.relative_to(ROOT).as_posix(),
            image_sha256=sha(image), donor_source_path=donor['source_path'],
            donor_prefix_last_s=7.0, donor_complete_pairs=4,
            selection='pre-existing three opened diagnostic drafts; not unseen test data',
            history_value_intervention='only features replaced; target metadata/mass/coords/tags unchanged',
            label_review='single_assistant_only; independent_admission_pending'))
        tasks.extend([
            dict(id=pair_id + ':old', pair_id=pair_id, role='old', question=old['question'],
                 options=old['options'], correct_index=labels[qid]['correct_index'],
                 label_origin='unchanged_official', evidence_role='old_question_not_proven_old_necessary'),
            dict(id=pair_id + ':current', pair_id=pair_id, role='current', question=question,
                 options=options, correct_index=0,
                 label_origin='single_assistant_development_annotation',
                 evidence_role='recent_four_committed_pairs')])
        review.append(dict(pair_id=pair_id, reviewer_identity=None, reviewer_role=None,
            decision='pending', current_sufficient_in_committed_input=None,
            old_answer_supported_before_query=None, old_evidence_intervals_s=None,
            recent_sufficient_for_old=None, old_evidence_repeats_afterwards=None,
            donor_unrelated_to_current_answer=None, donor_review_intervals_s=[0.0, 7.0],
            notes=None, manifest_sha256=None))
    for task in tasks:
        for arm in ARMS:
            for shift in range(5):
                schedule.append(dict(task_id=task['id'], pair_id=task['pair_id'], arm=arm, shift=shift,
                    content_scored=not (task['role'] == 'old' and arm == 'history_value_swap')))
    manifest = dict(schema='perception-interference-v1', pairs=pairs,
        resolution=[288, 384], sampling_fps=1, bank='LS4_4', visual_token_limit=864,
        recent_visual_token_limit=432, context_limit=8192, max_new_tokens=8,
        shifts=list(range(5)), semantic_questions=6, query_anchors=3, formal_answers=90,
        content_scored_answers=75, old_swap_answers_not_scored=15,
        expected_target_visual_forwards=sum(p['committed_pairs'] for p in pairs),
        donor_visual_forwards=12, main_hypothesis='specific historical values affect recent readout',
        scope='opened development diagnosis; not long-memory necessity or generalization benchmark',
        no_training=True, no_new_memory_compressor=True, no_default_changes=True,
        stop='one diagnostic round; no effect or ambiguous attribution ends this branch',
        GPU_authorized=False, independent_review_status='pending',
        proposed_GPU=dict(partition='g078t', comment='gx_llm', GPUs=1, CPUs=4,
            RAM_GiB=24, minutes=20, pretrained_loads=1, training_steps=0,
            downloads=0, automatic_retry=False, requeue=False),
        input_origin_sha256={
            str(p.relative_to(ROOT).as_posix()): sha(p) for p in (
                OLD / 'paired_current_drafts_LOCAL_ONLY.json', OLD / 'case_review_index_LOCAL_ONLY.json',
                selection / 'labels_LOCAL_ONLY.json', selection / 'videos.json')})
    save(INPUTS / 'manifest_LOCAL_ONLY.json', manifest)
    save(INPUTS / 'tasks_LOCAL_ONLY.json', tasks)
    save(INPUTS / 'schedule.json', schedule)
    for record in review:
        record['manifest_sha256'] = sha(INPUTS / 'manifest_LOCAL_ONLY.json')
    save(PREP / 'independent_review_LOCAL_ONLY.json', review)
    print(dict(status='prepared_not_admitted', pairs=3, semantic_questions=6,
        planned_answers=len(schedule), GPU_jobs=0))


if __name__ == '__main__':
    main()
