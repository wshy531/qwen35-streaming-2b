"""Select new declared StreamingBench files using frozen metadata only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/delayed_readiness_prepare_20261104_01'
SEED = 'delayed-read-confirm-20261104-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def order(value):
    return hashlib.sha256((SEED + value).encode()).hexdigest()


def seconds(value):
    parts = value.split(':')
    if len(parts) not in (2, 3):
        return None
    try:
        numbers = [float(x) for x in parts]
    except ValueError:
        return None
    if any(x < 0 for x in numbers) or any(x >= 60 for x in numbers[-2:]):
        return None
    return sum(x * 60 ** i for i, x in enumerate(reversed(numbers)))


def main():
    annotation = ROOT / 'research/dataset-survey-20261029/questions_real.json'
    prior_path = ROOT / 'research/dataset-survey-20261029/eval_annotation_proposal.json'
    index_path = ROOT / 'research/dataset-survey-20261030/zip_index_report.json'
    excluded = {PurePosixPath(x['video_path']).name for x in read(prior_path)['media']}
    members = {}
    for archive in read(index_path)['archives']:
        for entry in archive['entries']:
            path = PurePosixPath(entry['path'])
            if path.suffix == '.mp4' and '__MACOSX' not in path.parts:
                name = path.parent.name + '_real.mp4'
                assert name not in members
                members[name] = dict(archive=archive['archive'], member=entry)
    media = defaultdict(list)
    labels = {}
    issues = []
    for row_index, row in enumerate(read(annotation)):
        name = PurePosixPath(row['video_path']).name
        if name in excluded:
            continue
        for question_index, question in enumerate(row['questions']):
            time = seconds(question['time_stamp'])
            if time is None or time < 2:
                issues.append(dict(media=name, row=row_index, question=question_index, reason='invalid_or_too_early_timestamp'))
                continue
            identity = 'streamingbench:' + hashlib.sha256(f'{name}:{row_index}:{question_index}'.encode()).hexdigest()
            item = dict(id=identity, media_name=name, question=question['question'],
                        options=question['options'], task_type=question['task_type'], query_time_s=time,
                        original_timestamp=question['time_stamp'], original_annotation_row=row_index,
                        original_question_index=question_index, required_ability=question.get('required_ability'))
            media[name].append(item)
            answer = question['answer'].strip()
            assert answer in 'ABCDE'[:len(item['options'])]
            labels[identity] = ord(answer) - ord('A')
    strata = defaultdict(list)
    for name, questions in media.items():
        if len(questions) < 3 or name not in members:
            continue
        for task in {q['task_type'] for q in questions}:
            strata[task].append(name)
    assert len(strata) == 10
    quotas = {task: min(6, len(names)) for task, names in strata.items()}
    remaining = 60 - sum(quotas.values())
    for task in sorted(strata, key=lambda task: (-len(strata[task]), task))[:remaining]:
        quotas[task] += 1
    selected = []
    used = set()
    for task, names in sorted(strata.items(), key=lambda item: (len(item[1]), item[0])):
        names = [name for name in names if name not in used]
        quota = quotas[task]
        assert len(names) >= quota, task
        bins = defaultdict(list)
        for name in names:
            end = max(q['query_time_s'] for q in media[name])
            bins['short' if end < 180 else 'medium' if end < 600 else 'long'].append(name)
        for bucket in bins.values():
            bucket.sort(key=order)
        chosen = []
        while len(chosen) < quota:
            for band in ('short', 'medium', 'long'):
                if bins[band] and len(chosen) < quota:
                    chosen.append(bins[band].pop(0))
        for name in chosen:
            used.add(name)
            ordered = sorted(media[name], key=lambda q: order(q['id']))
            first = next(q for q in ordered if q['task_type'] == task)
            questions = [first, *[q for q in ordered if q['id'] != first['id']][:2]]
            selected.append(dict(media_name=name, selection_stratum=task,
                                 questions=questions, **members[name]))
    tasks = [q for entry in selected for q in entry['questions']]
    assert len(selected) == 60 and len(tasks) == 180
    assert not {x['media_name'] for x in selected} & excluded
    summary = dict(status='metadata_candidate_only', seed=SEED, videos=60, questions=180,
                   existing_declared_files_excluded=len(excluded), eligible_unseen_declared_files=len(media),
                   selection_quotas=quotas,
                   task_counts=dict(Counter(q['task_type'] for q in tasks)),
                   query_bins=dict(Counter('<180s' if q['query_time_s'] < 180 else
                                           '180-600s' if q['query_time_s'] < 600 else '>=600s' for q in tasks)),
                   compressed_bytes=sum(x['member']['compressed_bytes'] for x in selected),
                   original_bytes=sum(x['member']['bytes'] for x in selected),
                   inputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (annotation, prior_path, index_path)},
                   unresolved=['parent filming-source identity and cross-dataset overlap',
                               'media availability, SHA and actual duration', 'sampled evidence and query cutoff',
                               'current/history/cross-event requirement labels'],
                   model_outputs_used=False, download_started=False, gpu_started=False,
                   selection_rule='target six files per task, rare tasks first, deficits redistributed to largest pools; round-robin prefix bins, SHA order; three questions per file including the selection task')
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {'candidate_manifest.json': selected, 'candidate_requests.json': tasks,
               'candidate_labels_LOCAL_ONLY.json': {q['id']: labels[q['id']] for q in tasks},
               'candidate_summary.json': summary, 'candidate_timestamp_issues.json': issues}
    for name, data in outputs.items():
        with (OUT / name).open('x', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
