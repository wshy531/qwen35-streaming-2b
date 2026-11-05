"""Build local evaluation lists using annotations and media inventories only."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/eval_consolidation_20261105_01'
INPUTS = {}


def load(path):
    raw = (ROOT / path).read_bytes()
    INPUTS[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def save(name, data):
    (OUT / f'{name}.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rank(value):
    return hashlib.sha256(f'eval-consolidation-20261105-v1:{value}'.encode()).hexdigest()


def short_band(duration):
    return next((f'{a}-{b}' for a, b in [(10, 20), (20, 40), (40, 60), (60, 90)]
                 if a <= duration < b or duration == b == 90), None)


def stream_band(time):
    return '<180' if time < 180 else '180-600' if time < 600 else '>=600'


def file_signature(member):
    return int(member['bytes']), str(member['crc32']).lower().removeprefix('0x').zfill(8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scale = 'runs/evidence_scale_prepare_20261102_01/'
    prior = 'runs/memory_prune_panel150_prepare_20261103_01/'
    paired = 'runs/midprune_memory_prepare_20261102_01/'
    order = load('runs/training_order_main_prepare_20261104_02/panel_LOCAL_ONLY.json')
    next_base = 'runs/learned_prune_data_prepare_20261101_01/upstream/'
    vidor = load(next_base + 'map_vid_vidorID.json')
    next_train = set(order['train']) | {
        q['video_id'] for q in load('runs/learned_prune_main_prepare_20261101_01/train.json')}
    next_train.update(q['video_id'] for q in load(
        'runs/history_utility_prepare_20261104_01/train_manifest_LOCAL_ONLY.json'))
    next_train_sha = {order['groups'][v]['sha256'] for v in next_train if v in order['groups']}
    next_train_parents = {vidor[v] for v in next_train if v in vidor}

    bank = load('runs/router-bank-teacher-20261101-01/dataset_all_LOCAL_ONLY.json')
    contrast = load('runs/router_contrast_features_prepare_20261101_01/requests.json')
    selector_path = 'runs/evidence_selector_train_prepare_20261029_01/'
    selector = load(selector_path + 'long_sources.json')
    video_teacher = [q for q in load(selector_path + 'short_sources.json') if q['role'] == 'train']
    video_teacher_sha = {q['media_sha256'] for q in video_teacher}
    # Conservatively exclude all sources used to construct these teacher/selector materials.
    sember_supervision = {q['source_id'].split('_start_')[0] for q in bank + contrast}
    sember_supervision.update(q['parent_source'] for q in selector)
    sember_sha = {q['sha256'] for q in selector}

    old_main = [q for q in load(scale + 'panel.json') if q['dataset'] == 'StreamingBench']
    existing_media = load('runs/streamingbench_official_tail_20261031_01/complete_manifest.json')
    media_inventory = {q['name']: q for q in existing_media['files']}
    assert all(q['media_sha256'] == media_inventory[q['source_id']]['sha256'] for q in old_main)
    assert not {q['media_sha256'] for q in old_main} & (video_teacher_sha | sember_sha | next_train_sha)
    old_regression_ids = {q['id'] for q in load(prior + 'regression150.json')
                          if q['dataset'] == 'StreamingBench'}
    main_labels = load(scale + 'labels_LOCAL_ONLY.json')
    old_diagnosis = load(paired + 'questions.json')
    diagnosis_labels = load(paired + 'labels_LOCAL_ONLY.json')
    paired_media = {s['name']: s for s in load(paired + 'sources.json')}
    for q in old_diagnosis:
        q['dataset'] = 'S-EMBER'
        q['source_id'] = q['source_name'].removesuffix('.mp4')
        q['source_group'] = q['source_id'].split('_start_')[0]
        q['media_sha256'] = paired_media[q['source_name']]['media_sha256']
        q['training_overlap_flag'] = (
            q['source_group'] in sember_supervision or q['media_sha256'] in sember_sha)

    short_pools = defaultdict(list)
    for video_id in order['development']:
        group = order['groups'][video_id]
        if (video_id in next_train or group['sha256'] in next_train_sha | video_teacher_sha | sember_sha
                or vidor.get(video_id) in next_train_parents):
            continue
        duration = float(group['duration_s'])
        band = short_band(duration)
        if band:
            short_pools[band].append((video_id, group, duration))
    short_dev, short_labels, used_parents = [], {}, set()
    short_availability = {}
    for band in ['10-20', '20-40', '40-60', '60-90']:
        groups = sorted(short_pools[band], key=lambda g: rank(g[0]))
        short_availability[band] = len(groups)
        selected = []
        for item in groups:
            video_id = item[0]
            parent = vidor.get(video_id, video_id)
            if parent not in used_parents:
                used_parents.add(parent)
                selected.append(item)
            if len(selected) == 15:
                break
        counts = Counter()
        for pass_index in range(2):
            for video_id, group, duration in selected:
                if sum(counts.values()) == 15:
                    break
                eligible = [q for q in group['questions'] if q['id'] not in short_labels]
                if not eligible:
                    continue
                q = min(eligible, key=lambda q: (counts[q['type'][0]], rank(q['id'])))
                assert q['options'][q['answer_index']] == q['answer_text']
                short_dev.append(dict(id=q['id'], dataset='NExT-QA', source_id=video_id,
                                      source_group=vidor.get(video_id, video_id),
                                      type=q['type'], duration_s=duration, query_time_s=duration,
                                      length_bin=band, query_policy='full_clip_EOF',
                                      question=q['question'], options=q['options'],
                                      media_path=group['path'], media_sha256=group['sha256']))
                short_labels[q['id']] = q['answer_index']
                counts[q['type'][0]] += 1

    old_ids = {q['id'] for q in old_main}
    assert len(old_main) == len(old_ids) == 150 and len(old_regression_ids) == 75
    assert old_regression_ids <= old_ids
    assert len(old_diagnosis) == len(diagnosis_labels) == 18
    # Preserve the full historical D18 view. Exclude flagged sources from current selection.
    eligible_diag = [q for q in old_diagnosis if not q['training_overlap_flag']]
    development = old_main + short_dev + eligible_diag
    regression_ids = old_regression_ids | {q['id'] for q in short_dev + eligible_diag}
    regression = [q['id'] for q in development if q['id'] in regression_ids]
    assert len(regression) == len(regression_ids)
    assert len({q['id'] for q in development}) == len(development)
    dev_labels = {**{q['id']: main_labels[q['id']] for q in old_main}, **short_labels,
                  **{q['id']: diagnosis_labels[q['id']]['correct_index'] for q in eligible_diag}}
    assert len(dev_labels) == len(development)
    assert all(0 <= dev_labels[q['id']] < len(q['options']) for q in development)
    save('development', development)
    save('regression_ids', regression)
    save('development_labels_LOCAL_ONLY', dev_labels)
    save('streamingbench_dev_extension_ids', [q['id'] for q in old_main
                                               if q['id'] not in old_regression_ids])
    save('legacy_D18', old_diagnosis)
    save('legacy_D18_evidence_LOCAL_ONLY', diagnosis_labels)

    annotation = load('research/dataset-survey-20261029/questions_real.json')
    zip_index = load('research/dataset-survey-20261030/zip_index_report.json')
    prior_names = set(media_inventory)
    official_seconds = runpy.run_path(str(
        ROOT / 'experiments/2b-delayed-readiness-v1/prepare_candidates.py'))['seconds']
    archives = {}
    for archive in zip_index['archives']:
        for member in archive['entries']:
            path = PurePosixPath(member['path'])
            if path.suffix == '.mp4' and '__MACOSX' not in path.parts:
                archives[path.parent.name + '_real.mp4'] = dict(archive=archive['archive'], member=member)
    blocked = {file_signature(q) for q in existing_media['files']}
    pool = defaultdict(list)
    stream_labels = {}
    for index, row in enumerate(annotation):
        name = PurePosixPath(row['video_path']).name
        if name in prior_names or name not in archives or file_signature(archives[name]['member']) in blocked:
            continue
        for q_index, q in enumerate(row['questions']):
            time = official_seconds(q['time_stamp'])
            answer = q['answer'].strip()
            if time is None or time < 2 or len(answer) != 1 or answer not in 'ABCDE'[:len(q['options'])]:
                continue
            qid = 'streamingbench:' + hashlib.sha256(f'{name}:{index}:{q_index}'.encode()).hexdigest()
            pool[name].append(dict(id=qid, source_id=name, type=q['task_type'], query_time_s=time,
                                   question=q['question'], options=q['options'], length_bin=stream_band(time),
                                   annotation_row=index, question_index=q_index,
                                   required_ability=q.get('required_ability')))
            stream_labels[qid] = ord(answer) - ord('A')

    # Reserve separate media for diagnostics. Each pair still needs visual annotation.
    diagnosis_queue = []
    used_signatures = set()
    for name in sorted(pool, key=lambda n: (
            any(q['length_bin'] == '>=600' or q['type'] == 'Prospective Reasoning'
                for q in pool[n]), rank(n))):
        rows = sorted(pool[name], key=lambda q: q['query_time_s'])
        current = [q for q in rows if q['type'] in {
            'Action Recognition', 'Attribute Recognition', 'Object Recognition', 'Spatial Understanding'}]
        pairs = [(a, b) for a in current for b in current
                 if b['query_time_s'] - a['query_time_s'] >= 120]
        signature = file_signature(archives[name]['member'])
        if not pairs or signature in used_signatures:
            continue
        a, b = min(pairs, key=lambda p: rank(p[0]['id'] + p[1]['id']))
        diagnosis_queue.append(dict(source_id=name, early_anchor=a, current_anchor=b,
                                    proposed_query_time_s=b['query_time_s'],
                                    cross_time_candidate_slot=len(diagnosis_queue) < 20,
                                    status='pending_visual_evidence_and_label_review', **archives[name]))
        used_signatures.add(signature)
        if len(diagnosis_queue) == 30:
            break
    save('new_diagnosis_source_queue', diagnosis_queue)

    by_band = defaultdict(list)
    for rows in pool.values():
        for q in rows:
            by_band[q['length_bin']].append(q)
    diagnosis_signatures = set(used_signatures)
    new_main, types, source_counts = [], Counter(), Counter()
    for band in ('>=600', '180-600', '<180'):
        for _ in range(50):
            available = [q for q in by_band[band]
                         if file_signature(archives[q['source_id']]['member']) not in used_signatures]
            if not available:
                break
            q = min(available, key=lambda q: (types[q['type']], rank(q['id'])))
            new_main.append({**q, **archives[q['source_id']],
                             'status': 'pending_media_and_source_admission'})
            types[q['type']] += 1
            source_counts[q['source_id']] += 1
            used_signatures.add(file_signature(archives[q['source_id']]['member']))
    # Fill deficits with a second question on already selected media before
    # increasing the transfer footprint. Keep distinct-file counts visible.
    selected_ids = {q['id'] for q in new_main}
    for band in ('>=600', '180-600', '<180'):
        missing = 50 - sum(q['length_bin'] == band for q in new_main)
        for _ in range(missing):
            available = [q for q in by_band[band] if q['id'] not in selected_ids
                         and source_counts[q['source_id']] == 1]
            if not available:
                break
            q = min(available, key=lambda q: (types[q['type']], rank(q['id'])))
            new_main.append({**q, **archives[q['source_id']],
                             'status': 'pending_media_and_source_admission'})
            selected_ids.add(q['id'])
            types[q['type']] += 1
            source_counts[q['source_id']] += 1
    assert max(source_counts.values(), default=0) <= 2
    assert not old_ids & selected_ids
    assert not (blocked | diagnosis_signatures) & {
        file_signature(archives[n]['member']) for n in source_counts}
    save('new_streamingbench_candidates', new_main)
    save('new_streamingbench_labels_LOCAL_ONLY', {q['id']: stream_labels[q['id']] for q in new_main})

    # The official table supplies frame counts only. Use them to order probes,
    # never to assign a verified duration or admit a clip to the evaluation list.
    short_csv = next_base + 'train.csv'
    raw = (ROOT / short_csv).read_bytes()
    INPUTS[short_csv] = hashlib.sha256(raw).hexdigest()
    candidates = defaultdict(list)
    known_parents = {vidor[v] for v in order['groups'] if v in vidor}
    earlier_sources = {q['video_id'] for q in load('runs/learned_prune_main_prepare_20261101_01/sources.json')}
    known_parents.update(vidor[v] for v in earlier_sources if v in vidor)
    for q in csv.DictReader(raw.decode('utf-8-sig').splitlines()):
        video_id = q['video_id']
        if (video_id in order['groups'] or video_id in earlier_sources or video_id in next_train
                or video_id not in vidor or vidor[video_id] in known_parents):
            continue
        band = short_band(int(q['frame_count']) / 30)
        if band:
            candidates[band].append(q)
    probes = []
    probe_parents = set()
    for band, rows in sorted(candidates.items()):
        count = 0
        for q in sorted(rows, key=lambda q: rank(q['video_id'] + ':' + q['qid'])):
            parent = vidor[q['video_id']]
            if parent in probe_parents:
                continue
            probe_parents.add(parent)
            probes.append(dict(video_id=q['video_id'], source_group=parent, qid=q['qid'],
                               frame_count=int(q['frame_count']), probe_band_at_assumed_30fps=band,
                               type=q['type'], status='pending_actual_duration_and_media'))
            count += 1
            if count == 30:
                break
    save('new_short_probe_queue', probes)

    summary = dict(
        development=len(development), regression=len(regression),
        old_streamingbench=150, regression_streamingbench=75,
        old_streamingbench_sources=len({q['source_id'] for q in old_main}),
        short_development=len(short_dev), short_development_sources=len({q['source_group'] for q in short_dev}),
        short_development_bands=dict(Counter(q['length_bin'] for q in short_dev)),
        short_eligible_video_counts=short_availability,
        old_diagnosis_admitted=len(eligible_diag),
        old_diagnosis_flagged_ids=[q['id'] for q in old_diagnosis if q['training_overlap_flag']],
        nextqa_training_ids_screened=len(next_train),
        sember_supervision_parent_ids_screened=len(sember_supervision),
        videocon_teacher_training_file_records_screened=len(video_teacher),
        new_streamingbench_candidates=len(new_main),
        new_streamingbench_candidate_sources=len({q['source_id'] for q in new_main}),
        new_streamingbench_bands=dict(Counter(q['length_bin'] for q in new_main)),
        new_streamingbench_types=dict(types),
        new_streamingbench_archive_bytes=sum(archives[n]['member']['compressed_bytes'] for n in source_counts),
        diagnosis_source_candidates=len(diagnosis_queue),
        diagnosis_archive_bytes=sum(q['member']['compressed_bytes'] for q in diagnosis_queue),
        new_short_probe_bands=dict(Counter(q['probe_band_at_assumed_30fps'] for q in probes)),
        frozen_validation_admitted=0,
        outstanding=['media duration and SHA checks', 'cross-dataset filming-session identity',
                     'new diagnosis evidence and answer review', 'method/protocol freeze'],
        source_input_sha256=INPUTS,
    )
    save('summary', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_input_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
