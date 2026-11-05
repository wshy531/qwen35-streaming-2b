"""CPU input identity before any new GPU request. No model forward or weights."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import sys
import torch
from canonical_prompt import PROMPT_VERSION, question_prompt, rotate
from transition_common import ROOT, HERE, PREP, INPUTS, REFERENCE, read, save, sha


def tensor_sha(value):
    import hashlib
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CPU-only preparation required')
    torch.set_num_threads(1)
    reference = read(PREP / 'canonical_input_reference_LOCAL_ONLY.json')
    receipt = read(PREP / 'canonical_input_reference_receipt.json')
    assert sha(PREP / 'canonical_input_reference_LOCAL_ONLY.json') == receipt['sha256']
    assert reference['model_answers_included'] is False and len(reference['rows']) == 1020
    queries = {r['id']: r for r in read(PREP / 'canonical_requests_LOCAL_ONLY.json')}
    sys.path.insert(0, str(ROOT / 'experiments/2b-sember-v1'))
    from sember_common import runtime
    from sember_runtime import make_builder
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(ROOT / 'models/Qwen3.5-2B', local_files_only=True)
    builder = make_builder(env, processor)
    prefixes = {}
    for key, nodes in reference['nodes'].items():
        chunks = [builder.prefix_ids]
        for node in nodes:
            label = f"<{node['center']:.1f} seconds>" if node['exact_pair'] else f"<{node['begin']:.1f}-{node['end']:.1f} seconds, compressed memory>"
            chunks.append(builder.tokenize(label + processor.vision_start_token + processor.video_token*node['tokens'] + processor.vision_end_token))
        prefixes[key] = torch.cat(chunks, 1)
    reproduced = []
    for row in reference['rows']:
        query = queries[row['id']]
        rotated, mapping = rotate(query, row['shift'])
        text = question_prompt(rotated)
        poisoned = dict(query, correct_index=4, model_prediction='MUST_NOT_RENDER')
        assert question_prompt(rotate(poisoned, row['shift'])[0]) == text
        ids = torch.cat((prefixes[row['nodes_key']], builder.question_ids(text)), 1)
        value = tensor_sha(ids)
        assert value == row['input_ids_sha256'] and ids.shape[1] == row['input_tokens'], (row['id'], row['arm'], row['shift'])
        reproduced.append(dict(id=row['id'], arm=row['arm'], shift=row['shift'], input_ids_sha256=value))
    cases = read(INPUTS / 'cases_LOCAL_ONLY.json')
    tasks = read(INPUTS / 'tasks_LOCAL_ONLY.json')
    schedule = read(INPUTS / 'schedule.json')
    byid = {t['id']: t for t in tasks}
    assert len(cases) == 6 and len(tasks) == 18 and len(schedule) == 270
    assert sum(r['content_scored'] for r in schedule) == 240
    source_rows = read(INPUTS / 'sources_LOCAL_ONLY.json')
    assert len(source_rows) == 5
    frame_checks = 0
    pairs = []
    reviewed = {c['case_id']:c for c in read(PREP / 'assistant_visual_review_LOCAL_ONLY.json')['cases']}
    for case in cases:
        early, late, old = [byid[case['id']+suffix] for suffix in (':early:current', ':late:current', ':late:old')]
        assert early['question'] == late['question'] and early['options'] == late['options']
        assert early['correct_index'] != late['correct_index']
        assert old['correct_index'] == early['correct_index']
        assert early['query_time_s'] < late['recent_frames'][0]['pts_s']
        assert old['query_time_s'] == late['query_time_s']
        # The new exact recent cards cover both earlier evidence and later input;
        # upstream navigation pages need not be copied to the cluster again.
        for evidence in reviewed[case['id']]['evidence']:
            assert sha(ROOT / evidence['relative_path']) == evidence['sha256']
        for task in (early, late):
            assert task['recent_frames'][-1]['pts_s'] == task['committed_through_s'] == task['query_time_s']
            assert len(task['recent_frames']) == 8 and len(task['evidence_frames']) == 2
            assert task['evidence_frames'] == task['recent_frames'][-2:]
            assert all(f['pts_s'] <= task['query_time_s'] for f in task['recent_frames'])
            frame_checks += len(task['recent_frames'])
        pairs.append(dict(case_id=case['id'], early_query_s=early['query_time_s'], late_query_s=late['query_time_s'],
            state_gap_s=late['query_time_s']-early['query_time_s'], different_frozen_answers=True,
            same_question_and_options=True, historical_answer_has_later_cues=case['later_recent_cue']))
        for shift in range(5):
            assert question_prompt(rotate(early, shift)[0]) == question_prompt(rotate(late, shift)[0])
    layouts = []
    for task in tasks:
        for shift in range(5):
            rotated, mapping = rotate(task, shift)
            assert mapping.index(task['correct_index']) == (task['correct_index']-shift) % 5
            text = question_prompt(rotated)
            bad = deepcopy(task)
            bad['correct_index'] = (task['correct_index'] + 1) % 5
            assert question_prompt(rotate(bad, shift)[0]) == text
            ids = builder.question_ids(text)
            assert ids.shape[1] <= 512
            layouts.append(dict(task_id=task['id'], shift=shift, suffix_ids_sha256=tensor_sha(ids), suffix_tokens=ids.shape[1]))
    for bad in (
        dict(question='test', options=['A. one', 'two']),
        dict(question='test', options=['B. one', 'A. two']),
        dict(question='test', options=['A. ', 'B. two']),
    ):
        try:
            question_prompt(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid embedded structural labels accepted')
    source_files = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md')]
    result = dict(status='passed_CPU_canonical_inputs_and_panel_contract', prompt_version=PROMPT_VERSION,
        historical_text_inputs_reproduced=1020, historical_sources=18, historical_questions=102,
        old_input_ID_rows=reproduced, panel_sources=5, transition_cases=6, question_time_tasks=18,
        unique_question_wordings=len({t['question'] for t in tasks}),
        question_option_forms=len({(t['question'], tuple(t['options'])) for t in tasks}),
        planned_transition_answers=270, primary_content_answers=240,
        suffix_layouts=layouts, transition_pairs=pairs, checked_recent_frame_references=frame_checks,
        no_model_forward=True, GPU_jobs=0, pretrained_loads=0, capability_claim=False,
        reviewed_query_times_not_wallclock_runtime=True, all_later_cue_notes_retained=True,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(source_files)},
        input_sha256={p.name:sha(p) for p in INPUTS.iterdir() if p.is_file()})
    save(output, result)
    print({k:v for k,v in result.items() if k not in ('old_input_ID_rows','suffix_layouts','transition_pairs','source_sha256','input_sha256')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
