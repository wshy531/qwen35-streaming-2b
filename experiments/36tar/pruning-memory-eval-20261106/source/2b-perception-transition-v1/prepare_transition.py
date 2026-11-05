"""Freeze state changes from reviewed pixels, before new model answers exist."""
from collections import defaultdict
from transition_common import ROOT, HERE, PREP, INPUTS, SOURCE, ARMS, read, save, sha
from canonical_prompt import PROMPT_VERSION, question_prompt, rotate


CASES = [
    dict(id='T01', source_case='A02', times=[63, 75], answers=[1, 0],
        question='Based on the most recently completed frame pair, what are the camera wearer\'s hands interacting with?',
        old_question='During the complete frame pair at 62-63 seconds, what were the camera wearer\'s hands interacting with?',
        options=['An undivided white rectangular container on the shelf.', 'Compartmented ice-cube trays.',
                 'A smartphone.', 'A folded towel.', 'A desk lamp.'],
        evidence_pages=['A02_60_75/p00.png'], later_recent_cue='Ice trays remain visible on the background shelf; old evidence necessity unproven.',
        review_note='62-63: hands hold divided trays; 74-75: hand interacts with an undivided white rectangular container.'),
    dict(id='T02', source_case='A03', times=[157, 167], answers=[3, 1],
        question='Based on the most recently completed frame pair, which object is the camera wearer actively handling?',
        old_question='During the complete frame pair at 156-157 seconds, which object was the camera wearer actively handling?',
        options=['A blue plate.', 'A smartphone.', 'A gray laundry hamper.', 'A red detergent bottle.', 'A paper bag.'],
        evidence_pages=['A03_146_167/p00.png', 'A03_146_167/p01.png'], later_recent_cue='Red bottle remains on the washer behind the phone; old evidence necessity unproven.',
        review_note='156-157: manipulating the red bottle/cap; 166-167: phone is actively held/operated while bottle is in background.'),
    dict(id='T03', source_case='A06', times=[215, 231], answers=[0, 1],
        question='Based on the most recently completed frame pair, where is the metal water bottle?',
        old_question='During the complete frame pair at 214-215 seconds, where was the metal water bottle?',
        options=['In the camera wearer\'s hands.', 'On the wooden tray beside a plant and a framed picture.',
                 'In a kitchen sink.', 'On the floor.', 'Inside a shopping cart.'],
        evidence_pages=['A06_214_237/p00.png', 'A06_214_237/p01.png'], later_recent_cue='Bottle is still held in part of late recent window before placement; old evidence necessity unproven.',
        review_note='214-215: bottle held in hands. 230 in final 230-231 pair shows bottle on tray. Do not require target visible in last single frame.'),
    dict(id='T04', source_case='C06', times=[57, 75], answers=[3, 2],
        question='Based on the most recently completed frame pair, which item is being held up in the camera wearer\'s hands?',
        old_question='During the complete frame pair at 56-57 seconds, which item was being held up in the camera wearer\'s hands?',
        options=['A paper bag.', 'A water bottle.', 'A dark round plate.', 'A clear pot lid.', 'A smartphone.'],
        evidence_pages=['C06_0_107/p03.png', 'C06_0_107/p04.png'], later_recent_cue='No clear pot lid seen in sampled late 68-75 window; absence alone does not prove necessity.',
        review_note='56-57: clear pot lid held up; 74-75: dark blue round plate held up. Avoid inferring cupboard open/closed from missed transition.'),
    dict(id='T05', source_case='C02', times=[173, 185], answers=[3, 1],
        question='Based on the most recently completed frame pair, which object is the camera wearer actively handling?',
        old_question='During the complete frame pair at 172-173 seconds, which object was the camera wearer actively handling?',
        options=['A metal pot lid.', 'A brown paper shopping bag.', 'A smartphone.', 'An avocado.', 'A folded towel.'],
        evidence_pages=['C02_152_185/p01.png', 'C02_152_185/p02.png'], later_recent_cue='Avocado held again at 178-181 within late recent window; old evidence necessity unproven.',
        review_note='172-173: avocado in both hands; 184-185: handling brown paper bag. No event count labels added.'),
    dict(id='T06', source_case='C03', times=[177, 195], answers=[3, 4],
        question='Based on the most recently completed frame pair, which printed region is beside the pointing fingertip?',
        old_question='During the complete frame pair at 176-177 seconds, which printed region was beside the pointing fingertip?',
        options=['A smartphone screen.', 'A shopping cart handle.', 'No clear pointing gesture.',
                 'A green promotional strip containing a QR code.', 'A shelf price label.'],
        evidence_pages=['C03_168_227/p00.png', 'C03_168_227/p01.png'], later_recent_cue='QR strip remains in the wider scene; old evidence necessity unproven.',
        review_note='176-177: finger near green QR promotional strip, not necessarily touching QR itself. 194-195: finger near shelf price label. No claim of reading QR contents.'),
]


def main():
    if INPUTS.exists():
        raise FileExistsError('retain previous freeze; do not silently regenerate')
    panel = {r['case_id']: r for r in read(SOURCE / 'panel_LOCAL_ONLY.json')['rows']}
    audits = {r['media_sha256']: r for r in read(SOURCE / 'input_audit_LOCAL_ONLY.json')['sources']}
    source_rows = {}
    cases, tasks = [], []
    for case in CASES:
        source = panel[case['source_case']]
        audit = audits[source['media_sha256']]
        name = source['query']['source_path'].split('/')[-1]
        row = dict(case, source_path=source['query']['source_path'], media_sha256=source['media_sha256'],
            label_origin='assistant_reviewed_development', independent_review=False,
            formal_benchmark_admitted=False, selected_by_new_model_scores=False)
        evidence = []
        for relative in case['evidence_pages']:
            path = SOURCE / 'details_LOCAL_ONLY' / relative
            evidence.append(dict(relative_path=path.relative_to(ROOT).as_posix(), sha256=sha(path)))
        row['evidence_images'] = evidence
        frames = audit['sampled_trace']
        for stage, time_s, answer in zip(('early', 'late'), case['times'], case['answers']):
            prefix = [f for f in frames if f['pts_s'] <= time_s]
            assert len(prefix) % 2 == 0 and prefix[-1]['pts_s'] == time_s
            assert len(prefix[-8:]) == 8
            tasks.append(dict(id=case['id'] + ':' + stage + ':current', case_id=case['id'], role='current',
                stage=stage, source_name=name, query_time_s=time_s, committed_through_s=time_s,
                question=case['question'], options=case['options'], correct_index=answer,
                evidence_frames=prefix[-2:], recent_frames=prefix[-8:],
                label_origin='assistant_reviewed_development', independent_review=False))
        tasks.append(dict(id=case['id'] + ':late:old', case_id=case['id'], role='old', stage='late',
            source_name=name, query_time_s=case['times'][1], committed_through_s=case['times'][1],
            question=case['old_question'], options=case['options'], correct_index=case['answers'][0],
            evidence_interval_s=[case['times'][0]-1, case['times'][0]],
            label_origin='assistant_reviewed_development', independent_review=False,
            old_evidence_necessary=None, necessity_status='not_established', later_cue_note=case['later_recent_cue']))
        cases.append(row)
        if name not in source_rows:
            source_rows[name] = dict(name=name, source_path=source['query']['source_path'],
                media_sha256=source['media_sha256'], max_query_s=0, sampled_trace=frames)
        source_rows[name]['max_query_s'] = max(source_rows[name]['max_query_s'], case['times'][1])
    for row in source_rows.values():
        row['sampled_trace'] = [f for f in row['sampled_trace'] if f['pts_s'] <= row['max_query_s']]
        row['complete_pairs'] = len(row['sampled_trace']) // 2
    schedule = []
    for task in sorted(tasks, key=lambda r: (r['source_name'], r['query_time_s'], r['id'])):
        for shift in range(5):
            rotated, _ = rotate(task, shift)
            assert 'correct_index' not in rotated
            question_prompt(rotated)
            for arm in ARMS:
                schedule.append(dict(task_id=task['id'], arm=arm, shift=shift,
                    content_scored=not (task['role'] == 'old' and arm == 'history_value_swap')))
    assert len(tasks) == 18 and len(schedule) == 270 and len(source_rows) == 5
    save(INPUTS / 'cases_LOCAL_ONLY.json', cases)
    save(INPUTS / 'tasks_LOCAL_ONLY.json', tasks)
    save(INPUTS / 'schedule.json', schedule)
    save(INPUTS / 'sources_LOCAL_ONLY.json', list(source_rows.values()))
    save(INPUTS / 'protocol.json', dict(schema='perception-transition-development-v1', prompt_version=PROMPT_VERSION,
        cases=6, independent_source_files=5, question_time_tasks=18,
        unique_question_wordings=len({t['question'] for t in tasks}),
        question_option_forms=len({(t['question'], tuple(t['options'])) for t in tasks}),
        current_task_times=12, old_task_times=6, planned_transition_answers=270, primary_content_answers=240,
        raw_labels_changed=False, independent_human_review=False, use='second_and_final_development_diagnostic_round',
        fixed_weights=True, resolution=[288,384], sampling_fps=1, total_visual_tokens=864, recent_visual_tokens=432,
        arms=list(ARMS), shifts=list(range(5)), max_new_tokens=8, full_causal_prefix_required=True,
        maximum_target_pairs=sum(r['complete_pairs'] for r in source_rows.values()),
        formal_memory_necessity_established=False, no_new_model_outputs_seen=True,
        old_reference_phase_required=True, old_reference_answers=30, old_reference_gate='30/30 old choices must reproduce before new transition phase',
        action_gate='at least two distinct sources, each with >=2 rotations early/late readable in A, B reverting to early answer, C recovering late answer',
        stop='end this branch if gate not met; no prompt sweep or repeated replacement of cases', GPU_jobs=0,
        excluded_proposals=[dict(name='cabinet_closed_state',reason='key closure not unambiguous at 1 FPS'),
            dict(name='late_washer_finger_target',reason='finger absent at intended late checkpoint; no active pointing evidence')]))
    print(dict(status='development_panel_frozen', sources=5, cases=6, question_time_tasks=18,
               transition_answers=270, primary_scored=240, GPU_jobs=0))


if __name__ == '__main__':
    main()
