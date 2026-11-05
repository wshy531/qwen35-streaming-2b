"""Real sampled pixels from review cards + random tiny model; zero real weights."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
from types import SimpleNamespace
from unittest.mock import patch
import sys
import torch
from PIL import Image
from common import ROOT, HERE, PREP, INPUTS, OLD, ARMS, read, save, sha, displayed
from interference_execution import exact_pairs, project, append, ask
from sember_common import runtime
from sember_banks import Bank
from sember_runtime import make_builder
from sember_protocol import prompt
from analyze import analyze_records


def tiles(path, trace, top):
    import hashlib
    image = Image.open(path).convert('RGB')
    result = {}
    for index, record in enumerate(trace):
        x, y = index % 4 * 384, top + index // 4 * 312
        tile = image.crop((x, y, x + 384, y + 288))
        assert hashlib.sha256(tile.tobytes()).hexdigest() == record['mid_rgb_sha256']
        result[record['frame_id']] = tile
    return result


def references():
    index = {r['query']['id']: r for r in read(OLD / 'case_review_index_LOCAL_ONLY.json')}
    result = []
    for pair in read(INPUTS / 'manifest_LOCAL_ONLY.json')['pairs']:
        row = index[pair['old_question_id']]
        for arm in ('recent', 'full'):
            for shift, choice in enumerate(row['metrics']['LS4_4/' + arm]['content_choice_original']):
                result.append(dict(task_id=pair['pair_id'] + ':old', arm=arm, shift=shift,
                                   expected_content_choice_original=choice))
    assert len(result) == 30
    return result


def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CPU only')
    torch.set_num_threads(1)
    torch.manual_seed(20261027)
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    audits = {r['pair_id']: r for r in read(PREP / 'cluster_media_CPU.json')['sources']}
    one = manifest['pairs'][0]
    a = audits[one['pair_id']]
    trace = a['donor_trace'] + a['old_review_trace'] + one['recent_frames']
    images = tiles(ROOT / a['donor_card']['relative_path'], a['donor_trace'], 56)
    images.update(tiles(ROOT / a['old_card']['relative_path'], a['old_review_trace'], 56))
    images.update(tiles(ROOT / one['image_relative_path'], one['recent_frames'], 24))
    assert len(images) == len(trace) == 22
    class FakeContainer:
        def __enter__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(codec_context=SimpleNamespace(thread_count=None))])
            return self
        def __exit__(self, *args):
            return False
        def decode(self, stream):
            for index in range(trace[-1]['frame_id'] + 1):
                yield SimpleNamespace(time=index / 12, to_image=lambda index=index: images[index].copy())
    with patch('av.open', return_value=FakeContainer()):
        pairs = list(exact_pairs(Path('CPU-fixture-not-a-real-video.mp4'), trace))
    assert len(pairs) == 11
    corrupted = deepcopy(trace)
    corrupted[0]['mid_rgb_sha256'] = '0' * 64
    with patch('av.open', return_value=FakeContainer()):
        try:
            list(exact_pairs(Path('CPU-corruption-fixture.mp4'), corrupted))
        except ValueError:
            pass
        else:
            raise AssertionError('corrupted sampled pixels were accepted')
    env = runtime()
    processor = env['AutoProcessor'].from_pretrained(ROOT / 'models/Qwen3.5-2B', local_files_only=True)
    builder = make_builder(env, processor)
    model = env['Qwen3_5ForConditionalGeneration'](
        env['Qwen3_5Config'](**env['tiny_config'](processor), attn_implementation='sdpa')).float().eval()
    # Source 1's real first four pairs serve only as a CPU interface donor fixture.
    donors = [project(env, model, builder, first, second)[0] for first, second in pairs[:4]]
    bank = Bank('LS', 108)
    descriptor = None
    for index, (first, second) in enumerate(pairs):
        value, _ = project(env, model, builder, first, second)
        descriptor = append(bank, value, index, first, second, descriptor)
    answers = []
    for arm in (*ARMS, 'recent'):
        result = ask(env, model, builder, bank, 'What is visible now?', arm, donors, max_new_tokens=2)
        assert result['branch_released'] and result['bank_unchanged']
        answers.append(dict(arm=arm, generated_ids=result['generated_ids'], input_tokens=result['input_tokens'],
            input_ids_sha256=result['input_ids_sha256'], position_ids_sha256=result['position_ids_sha256']))
    assert answers[0] == answers[-1]
    import importlib.util
    worker_spec = importlib.util.spec_from_file_location('interference_GPU_worker_under_test', HERE / 'evaluate.py')
    evaluate = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(evaluate)
    forbidden_output = ROOT / 'runs/CPU_must_not_start_GPU_interference'
    assert not forbidden_output.exists()
    with patch.dict(os.environ, {'SLURM_JOB_ID': '', 'SLURM_CPUS_PER_TASK': '4'}):
        try:
            evaluate.main(forbidden_output, PREP / 'not_a_real_execution_lock.json', Path('not_a_media_directory'))
        except RuntimeError as error:
            assert 'Slurm' in str(error)
        else:
            raise AssertionError('real model worker accepted a non-Slurm invocation')
    assert not forbidden_output.exists()
    tasks = read(INPUTS / 'tasks_LOCAL_ONLY.json')
    task_byid = {t['id']: t for t in tasks}
    schedule = read(INPUTS / 'schedule.json')
    refs = references()
    expected = {(r['task_id'], r['arm'], r['shift']): r['expected_content_choice_original'] for r in refs}
    raw = []
    for item in schedule:
        task = task_byid[item['task_id']]
        choice = expected.get((item['task_id'], item['arm'], item['shift']), task['correct_index'])
        text = 'unknown' if choice is None else chr(65 + (choice-item['shift']) % 5)
        raw.append(dict(**item, text=text))
    analysis = analyze_records(raw, tasks, schedule, read(PREP / 'assistant_review_LOCAL_ONLY.json'), refs)
    assert analysis['old_reference_reproduction']['passed']
    assert sum(r['denominator'] for r in analysis['primary']['totals'].values()) == 65
    poisoned = deepcopy(raw)
    first = poisoned[0]
    old_choice = expected[(first['task_id'], first['arm'], first['shift'])]
    first['text'] = chr(65 + ((0 if old_choice is None else old_choice+1)-first['shift']) % 5)
    assert analyze_records(poisoned, tasks, schedule, read(PREP / 'assistant_review_LOCAL_ONLY.json'), refs)['decision'] == 'stop_baseline_reproduction_mismatch'
    reference_path = PREP / 'old_reference_predictions_LOCAL_ONLY.json'
    if reference_path.exists():
        assert read(reference_path) == refs
    else:
        save(reference_path, refs)
    source = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md', '.sh')]
    save(output, dict(status='passed_CPU_shared_execution', real_sampled_pixels=22,
        synthetic_decoder_fixture=True, real_video_decoder_not_exercised_here=True,
        fixture_is_gapped_not_actual_full_query_history=True,
        corrupted_RGB_rejected=True, tiny_visual_forwards=15, tiny_QA=answers,
        actual_pretrained_loads=0, GPU_jobs=0, no_accuracy_or_speed_claim=True,
        analysis_primary_count=65, baseline_mismatch_stops=True,
        non_Slurm_GPU_worker_rejected_before_loading=True,
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(source)}))
    print(dict(status='passed_CPU_shared_execution', real_sampled_pixels=22,
        tiny_visual_forwards=15, tiny_QA=4, formal_model_answers=0, GPU_jobs=0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
