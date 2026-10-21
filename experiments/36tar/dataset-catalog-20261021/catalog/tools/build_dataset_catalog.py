"""Inventory existing assets and frozen questions. No download/model/GPU/remote writes."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/dataset_catalog_20261021_01'
CAT = OUT / 'catalog'
SOURCES = {}
TASKS = {'CW': '原因/动机（作者CW）', 'CH': '方式/如何（作者CH）',
         'TN': '后续动作（作者TN）', 'TP': '之前动作（作者TP）', 'TC': '时间关系（作者TC）',
         'EPM': '历史事件/物体回忆', 'ASI': '动作先后', 'HLD': '无关问题/幻觉检测',
         'manual_order': '人工事件先后', 'manual_detail': '人工旧画面细节', 'anchored_location': '人工时间锚定位置'}
BINS = [('lt30', '不足30秒', 0, 30), ('30to60', '30—不足60秒', 30, 60),
        ('60to180', '1—不足3分钟', 60, 180), ('180to300', '3—不足5分钟', 180, 300),
        ('300to600', '5—不足10分钟', 300, 600), ('ge600', '至少10分钟', 600, float('inf'))]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(name):
    p = ROOT / name
    SOURCES[name] = dict(bytes=p.stat().st_size, sha256=sha(p))
    return json.loads(p.read_text(encoding='utf-8'))


def save(name, value):
    p = CAT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def binof(seconds):
    if seconds is None:
        return 'unknown'
    return next(key for key, _, lo, hi in BINS if lo <= seconds < hi)


def counts(rows, key):
    return dict(sorted(Counter(key(row) for row in rows).items()))


def stats(rows):
    return dict(videos=len(rows), bytes=sum(v['bytes'] for v in rows),
        min_seconds=min((v['duration_s'] for v in rows), default=None),
        max_seconds=max((v['duration_s'] for v in rows), default=None),
        sum_seconds=sum(v['duration_s'] for v in rows), duration_bins=counts(rows, lambda v: v['duration_bin']))


def main():
    import av  # Local container-header reading only; no frame decoding or inference.
    CAT.mkdir(parents=True, exist_ok=True)
    assert not any(CAT.iterdir()), 'empty output directory required'
    old_media = read('runs/2b_collaboration_sync_20261020_01/media_manifest124_LOCAL.json')
    new_media = read('runs/2b_validation_T_prepare_20261020_01/download_manifest.json')['files']
    media_sources = {}
    identity = {}
    for item in old_media:
        rel = item['local_source_path']
        media_sources[rel] = dict(dataset=item['dataset'], source_id=item['source_id'],
            batch='initial124', cluster_preferred='data/2b-memory-eval-20261020/' + item['relative_path'],
            expected_sha256=item['sha256'], expected_bytes=item['bytes'],
            source_path=item['original_source_path'])
    for item in new_media:
        media_sources[item['media_path']] = dict(dataset='nextgqa', source_id=item['source_id'],
            batch='T-new40', cluster_preferred='data/T-new40-20261020/' + item['source_path'] + '.mp4',
            expected_sha256=item['sha256'], expected_bytes=item['bytes'], source_path=item['source_path'])
    all_assets = []
    for p in sorted((ROOT / 'datasets').rglob('*')):
        video = p.suffix.lower() in {'.mp4', '.webm', '.mkv', '.avi', '.mov'}
        photo = p.suffix.lower() in {'.jpg', '.jpeg'} and p.parent.name in {'coat_controls_v1', 'readout_holdout_v1'}
        if not p.is_file() or not (video or photo):
            continue
        rel = p.relative_to(ROOT).as_posix()
        h = sha(p)
        base = dict(id='sha256:' + h, sha256=h, bytes=p.stat().st_size, local_path=rel,
                    asset_type='video' if video else 'static_photo', batch=p.parent.name,
                    dataset='legacy', source_id=rel, source_path=rel,
                    cluster_preferred='data/legacy-architecture-20261021/' + p.relative_to(ROOT / 'datasets').as_posix())
        if rel in media_sources:
            base.update(media_sources[rel])
            assert h == base.pop('expected_sha256') and base['bytes'] == base.pop('expected_bytes')
            identity[base['dataset'], base['source_id']] = base['id']
            base['material_kind'] = 'simulated_3d_environment' if base['source_path'].startswith('OpenEQA/') else 'natural_video'
        elif p.parent.name in {'baseline_synthetic', 'stability_synthetic', 'window_synthetic'}:
            base['material_kind'] = 'synthetic_mechanism_control'
        elif p.parent.name.startswith('history_integration') or p.name == 'hands_vfr.mp4':
            base['material_kind'] = 'derived_mechanism_control'
        else:
            base['material_kind'] = 'static_control' if photo else 'natural_video'
        if video:
            with av.open(str(p)) as container:
                s = container.streams.video[0]
                duration = float(s.duration * s.time_base) if s.duration is not None else container.duration / 1000000
                base.update(duration_s=duration, duration_bin=binof(duration),
                    duration_method='video_stream_duration' if s.duration is not None else 'container_header_duration',
                    width=s.width, height=s.height, average_rate=str(s.average_rate),
                    duration_is_container_metadata_not_new_PTS_audit=True)
        else:
            base.update(duration_s=None, duration_bin='not_applicable')
        all_assets.append(base)
    assert len(media_sources) == 164 and len(identity) == 164
    assert len({a['id'] for a in all_assets}) == len(all_assets), 'duplicate bytes need explicit alias grouping'
    asset_map = {a['id']: a for a in all_assets}
    pool = read('runs/2b_collaboration_sync_20261020_01/candidate_pool385.json')
    tpool = read('runs/2b_validation_T_prepare_20261020_01/candidate_questions119.json')
    original235 = read('experiments/2b-memory-eval/data/candidate_manifest.json')
    initial_ids = {q['id'] for q in original235}
    releases = []
    questions = {}
    for row in pool + tpool:
        uid = row['id']
        assert uid not in questions
        media_id = identity[row['dataset'], row['source_id']]
        a = asset_map[media_id]
        batch = 'T-new40' if row in tpool else 'initial235' if uid in initial_ids else 'expansion78' if row['split'] == 'comparison' else 'new24_72'
        intervals = row.get('evidence_intervals')
        gap = max(0, row['query_time_s'] - max(end for _, end in intervals)) if intervals else None
        questions[uid] = dict(id=uid, semantic_question_group=uid, media_id=media_id,
            dataset=row['dataset'], origin='official', batch=batch, source_path=row['source_path'],
            question=row['question'], options=row['options'], answer_index=row['answer_index'],
            official_task=row['task'], task_name=TASKS[row['task']],
            reviewed_type=None, review_status='candidate_not_admitted_by_this_catalog',
            historical_split=row['split'], duration_s=a['duration_s'], duration_bin=a['duration_bin'],
            author_duration_s=row.get('duration_s'), query_time_s=row['query_time_s'],
            query_prefix_duration_bin=binof(row['query_time_s']), evidence_intervals=intervals,
            evidence_gap_proxy_s=gap, old_evidence_necessity_proven=False,
            independent_second_review=False, memberships=[])
    def membership(name, rows, origin, reviewed=False):
        ids = [q['id'] for q in rows]
        assert len(ids) == len(set(ids)) and set(ids) <= questions.keys(), name
        for uid in ids:
            questions[uid]['memberships'].append(name)
        media_ids = sorted({questions[uid]['media_id'] for uid in ids})
        releases.append(dict(name=name, questions=len(ids), media_files=len(media_ids), ids=ids, media_ids=media_ids,
            origin=origin, reviewed=reviewed, additive_to_other_releases=False,
            durations=stats([asset_map[i] for i in media_ids]),
            task_counts=counts([questions[i] for i in ids], lambda q: q['official_task']),
            reviewed_types=counts([questions[i] for i in ids], lambda q: q['reviewed_type'] or 'not_reclassified')))
    natural = read('runs/2b_natural_eval_20261020_01/scorer_rows_local.json')
    for row in natural:
        questions[row['id']].update(review_status='single_reviewer_admitted_and_evaluated_2B171',
            reviewed_type=row['curation']['stratum'], reviewer_note=row.get('review', {}).get('note', row['curation'].get('note')))
    dispositions = read('runs/2b_validation_T_prepare_20261020_01/content_dispositions119.json')
    for row in dispositions:
        questions[row['id']].update(review_status='T_' + row['content_verdict'],
            reviewed_type=row.get('stratum'), reviewer_note=row.get('note'),
            memory_character=row.get('memory_character'))
    long_labels = {q['id']: q for q in read('runs/2b_long_stream_prepare_20261020_01/stream_labels34_LOCAL_ONLY.json')}
    long_rows = read('runs/2b_long_stream_prepare_20261020_01/stream_requests34.json')
    ovo_original = {f"ovo:{q['id']}": q for q in read('experiments/2b-memory-eval/data/upstream/ovo/ovo_bench_new.json')}
    for row in long_rows:
        if row['id'] in questions or not row['id'].startswith('ovo:'):
            continue
        original = ovo_original[row['id']]
        assert row['question'] == original['question'] and row['options'] == original['options']
        assert long_labels[row['id']]['answer_index'] == original['gt']
        media_id = identity['ovo', row['source_path']]
        a = asset_map[media_id]
        questions[row['id']] = dict(id=row['id'], semantic_question_group=row['id'], media_id=media_id,
            dataset='ovo', origin='official', batch='longstream_official_extra', source_path=row['source_path'],
            question=row['question'], options=row['options'], answer_index=original['gt'], official_task=original['task'],
            task_name=TASKS[original['task']], reviewed_type=None, review_status='longstream_' + long_labels[row['id']]['verdict'],
            historical_split='open_longstream_development', duration_s=a['duration_s'], duration_bin=a['duration_bin'],
            query_time_s=row['query_time_s'], query_prefix_duration_bin=binof(row['query_time_s']),
            evidence_intervals=None, evidence_gap_proxy_s=None, old_evidence_necessity_proven=False,
            independent_second_review=False, memberships=[])
    manual = read('runs/2b_long_stream_prepare_20261020_01/manual6_delayed_questions_LOCAL_ONLY.json')
    for row in manual:
        media_id = identity['ovo', row['source_path']]
        task = 'manual_order' if row['id'][-2:] in {'01', '02', '03', '04'} else 'manual_detail'
        questions[row['id']] = dict(id=row['id'], semantic_question_group=row['id'], media_id=media_id,
            dataset='manual_on_OVO', origin='manual', batch='longstream_manual6', source_path=row['source_path'],
            question=row['question'], options=row['options'], answer_index=row['answer_index'], official_task=task,
            task_name=TASKS[task], reviewed_type='T' if task == 'manual_order' else 'F',
            review_status='single_reviewer_open_development', duration_s=asset_map[media_id]['duration_s'],
            duration_bin=asset_map[media_id]['duration_bin'], query_time_s=row['query_time_s'],
            query_prefix_duration_bin=binof(row['query_time_s']), evidence_gap_proxy_s=row.get('witness_gap_s'),
            witnesses=row['witnesses'], old_evidence_necessity_proven=False, independent_second_review=False, memberships=[])
    early = read('runs/memory_task_alignment_20261019_01/diagnostic9_cases_scorer_only.json')
    for row in early:
        if row['id'] in questions:
            continue
        assert row['id'].startswith('derived:ovo69:')
        media_id = identity['ovo', row['source_id']]
        questions[row['id']] = dict(id=row['id'], semantic_question_group='derived:ovo69:anchored_location',
            media_id=media_id, dataset='manual_on_OVO', origin='derived_query_time', batch='early_onion_3checkpoints',
            source_path=row['source_path'], question=row['question'], options=row['options'], answer_index=row['answer_index'],
            official_task='anchored_location', task_name=TASKS['anchored_location'], reviewed_type='F',
            review_status='single_event_diagnostic_not_independent_questions', duration_s=asset_map[media_id]['duration_s'],
            duration_bin=asset_map[media_id]['duration_bin'], query_time_s=row['query_time_s'],
            query_prefix_duration_bin=binof(row['query_time_s']), evidence_intervals=row.get('evidence_intervals'),
            old_evidence_necessity_proven=False, independent_second_review=False, memberships=[])
    for row in long_rows:
        questions[row['id']]['longstream_review'] = long_labels[row['id']]
        if row['id'].startswith('ovo:'):
            questions[row['id']]['review_status'] = 'longstream_' + long_labels[row['id']]['verdict']
    membership('最初100视频235候选', original235, 'experiments/2b-memory-eval/data/candidate_manifest.json')
    membership('补充78题_复用原比较视频', [q for q in pool if q['id'] not in initial_ids and q['split'] == 'comparison'], 'candidate_pool385.json')
    membership('新增24视频72候选', [q for q in pool if q['split'] == 'future_holdout_candidate'], 'candidate_pool385.json')
    membership('2B首轮67视频171题', natural, '2b_natural_eval_20261020_01', True)
    membership('T新40视频119候选', tpool, '2b_validation_T_prepare_20261020_01')
    tadmitted = read('runs/2b_validation_T_prepare_20261020_01/scorer_rows_LOCAL_ONLY.json')
    membership('T准入39视频98题', tadmitted, '2b_validation_T_prepare_20261020_01', True)
    membership('长视频6源34中途提问', long_rows, '2b_long_stream_prepare_20261020_01', True)
    membership('人工长期开发6题_34的子集', manual, 'manual6_delayed_questions_LOCAL_ONLY.json', True)
    membership('原EPM6题加同一洋葱3截点', early, 'memory_task_alignment_20261019_01', True)
    membership('2B首轮早期9题_171的子集', read('runs/2b_early9_diagnostic_20261020_01/requests.json'), '2b_early9_diagnostic_20261020_01', True)
    for folder, name in [('08b_frozen74_compare_20261019_01', '0.8B开放74题'),
                         ('08b_holdout47_compare_20261019_01', '0.8B已打开47题'),
                         ('08b_causal_memory_dev_20261019_01', '0.8B开发24题')]:
        membership(name, read('runs/' + folder + '/all_questions.json'), folder, True)
    # Keep the earliest 15 frozen window checks separate from benchmark questions.
    legacy15 = read('datasets/natural_commons_v1/cases.json')
    legacy_checks = []
    legacy_map = {p.stem: a['id'] for a in all_assets for p in [ROOT / a['local_path']] if p.parent.name == 'natural_commons_v1'}
    for i, row in enumerate(legacy15['cases']):
        legacy_checks.append(dict(id='legacy_natural15:' + str(i + 1), media_id=legacy_map[row['video']], **row))
    planned = read('runs/2b_memory_value_prepare_20261021_01/media_manifest1920.json')
    selection = read('runs/2b_memory_value_prepare_20261021_01/source_selection_receipt.json')
    raw_audit = read('experiments/2b-memory-eval/data/audit_summary.json')
    videos = [a for a in all_assets if a['asset_type'] == 'video']
    official = [q for q in questions.values() if q['origin'] == 'official']
    summary = dict(snapshot_date='2026-10-21', unit='physical video file or question+query-time item; not independent filming',
        media=stats(videos), unique_media_sha256=len({a['sha256'] for a in videos}),
        extensions=counts(videos, lambda v: Path(v['local_path']).suffix),
        video_kinds={kind: stats([v for v in videos if v['material_kind'] == kind]) for kind in sorted({v['material_kind'] for v in videos})},
        static_photos=len(all_assets) - len(videos), static_photo_bytes=sum(a['bytes'] for a in all_assets if a['asset_type'] == 'static_photo'),
        official_benchmark_videos=stats([v for v in videos if v['dataset'] in {'nextgqa', 'ovo'}]),
        all_question_time_items=len(questions), semantic_question_groups=len({q['semantic_question_group'] for q in questions.values()}),
        official_questions=len(official), official_task_counts=counts(official, lambda q: q['official_task']),
        official_duration_bins=counts(official, lambda q: q['duration_bin']),
        official_task_by_duration={task: counts([q for q in official if q['official_task'] == task], lambda q: q['duration_bin']) for task in TASKS if any(q['official_task'] == task for q in official)},
        earliest_frozen_natural_checks=len(legacy_checks),
        question_rows_not_all_admitted=True, difficulty='not re-rated by this catalog; author type and reviewed F/T/C are distinct',
        planned=dict(status='paused_not_downloaded', video_clips=len(planned), bytes=sum(p['bytes'] for p in planned),
            sessions=4, target_questions=96, admitted_questions=0, actual_duration_verified=False,
            note='4 approximately4h targets with recording gaps, same household; not four uncut independent environments'),
        raw_annotation_pool=dict(nextgqa_rows=raw_audit['upstream_nextgqa_rows'], invalid_grounding=3,
            ovo_all_tasks=raw_audit['upstream_ovo_rows'], ovo_selected_task_rows=sum(raw_audit['upstream_ovo_tasks'][t] for t in ('EPM','ASI','HLD'))),
        scope='all source media under project datasets; run thumbnails/source_snapshot copies excluded')
    save('media.json', all_assets)
    save('questions.json', list(questions.values()))
    save('releases.json', releases)
    save('legacy_checks15.json', legacy_checks)
    save('summary.json', summary)
    save('planned1920_NOT_DOWNLOADED.json', dict(status='paused', media=planned, source_selection_receipt=selection))
    save('provenance.json', SOURCES)
    print(json.dumps(dict(video_files=len(videos), total_bytes=summary['media']['bytes'], photo_files=summary['static_photos'],
        official_questions=len(official), question_time_items=len(questions), semantic_groups=summary['semantic_question_groups'],
        official_task_counts=summary['official_task_counts'], releases=[(x['name'],x['media_files'],x['questions']) for x in releases]),ensure_ascii=False))


if __name__ == '__main__':
    main()
