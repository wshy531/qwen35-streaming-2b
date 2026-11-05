"""Verify the received media and freeze the source-screened short-video panel."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import PurePosixPath

import av

from build import rank, short_band
from fetch_validation import ROOT, OUT, PLAN, read, save

PRIOR = ROOT / 'runs/eval_consolidation_20261105_01'
NEXT = ROOT / 'runs/learned_prune_data_prepare_20261101_01/upstream'
BANDS = ('10-20', '20-40', '40-60', '60-90')


def short_provenance():
    source_map = read(NEXT / 'map_vid_vidorID.json')
    trained = read(ROOT / 'runs/training_order_main_prepare_20261104_02/panel_LOCAL_ONLY.json')
    known_ids = set(trained['groups'])
    hashes = {g['sha256'] for g in trained['groups'].values()}
    earlier = read(ROOT / 'runs/learned_prune_main_prepare_20261101_01/sources.json')
    known_ids.update(q['video_id'] for q in earlier)
    hashes.update(q['mirror_lfs_sha256'] for q in earlier)
    for path in sorted((ROOT / 'datasets/2b-memory-eval').glob('nextgqa*/download_manifest.json')):
        for row in read(path)['files']:
            known_ids.add(PurePosixPath(row.get('source_path', '')).stem)
            if row.get('sha256'):
                hashes.add(row['sha256'])
    development = read(PLAN / 'development.json')
    known_ids.update(q['source_id'] for q in development if q['dataset'] == 'NExT-QA')
    hashes.update(q['media_sha256'] for q in development)
    teacher = ROOT / 'runs/evidence_selector_train_prepare_20261029_01'
    hashes.update(q['sha256'] for q in read(teacher / 'long_sources.json'))
    hashes.update(q['media_sha256'] for q in read(teacher / 'short_sources.json') if q['role'] == 'train')
    hashes.update(r['sha256'] for r in read(
        ROOT / 'runs/streamingbench_official_tail_20261031_01/complete_manifest.json')['files'])
    probe_used = read(PLAN / 'excluded_from_future_holdout.json')
    known_ids.update(probe_used['video_ids'])
    hashes.update(probe_used['sha256s'])
    groups = {source_map[i] for i in known_ids if i in source_map}
    groups.update(probe_used['source_groups'])
    return source_map, known_ids, groups, hashes


def valid_options(q):
    opts = [q[f'a{i}'] for i in range(5)]
    return all(o.strip() for o in opts) and len(set(opts)) == 5 and q['answer'] in opts


def media_record(receipt, full_decode=False, query_times=()):
    path = ROOT / receipt['media_path']
    assert path.stat().st_size == receipt['bytes']
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            sha.update(chunk)
    assert sha.hexdigest() == receipt['sha256']
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        video.thread_count = 1
        assert video.duration is not None or container.duration is not None
        duration = float(video.duration * video.time_base) if video.duration is not None else container.duration / 1e6
        start = float((video.start_time or 0) * video.time_base)
        count = 0
        previous_time = None
        if full_decode:
            for frame in container.decode(video):
                assert frame.time is not None
                assert previous_time is None or frame.time > previous_time
                previous_time = frame.time
                count += 1
        bounds = {}
        for query in sorted(set(query_times)):
            if not 0 <= query <= duration:
                bounds[str(query)] = None
                continue
            container.seek(int((start + max(0, query - 2)) / video.time_base), stream=video, backward=True)
            last = None
            for frame in container.decode(video):
                if frame.time is None:
                    continue
                relative = frame.time - start
                if relative > query:
                    break
                last = relative
            bounds[str(query)] = last
        return dict(duration_s=duration, decoded_frames=count if full_decode else None,
                    width=video.codec_context.width, height=video.codec_context.height,
                    near_cutoff_pts=bounds)


def build_short(complete):
    probes = {r['video_id']: r for r in read(PLAN / 'short_validation_probe_queue.json')}
    source_map, known_ids, known_groups, known_hashes = short_provenance()
    questions = defaultdict(list)
    with (NEXT / 'train.csv').open(encoding='utf-8-sig', newline='') as f:
        for q in csv.DictReader(f):
            if q['video_id'] in probes and valid_options(q):
                questions[q['video_id']].append(q)
    candidates = defaultdict(list)
    verified_groups, verified_hashes, audit = set(), set(), []
    for media in sorted(complete, key=lambda r: rank(r['name'])):
        video_id = PurePosixPath(media['name']).stem
        probe = probes[video_id]
        group = source_map[video_id]
        reasons = []
        try:
            info = media_record(media, full_decode=True)
        except (av.FFmpegError, AssertionError, OSError, ValueError, IndexError) as error:
            audit.append(dict(media, eligible=False, reasons=[type(error).__name__]))
            continue
        band = short_band(info['duration_s'])
        if band is None:
            reasons.append('outside_duration_bands')
        if info['decoded_frames'] != probe['frame_count']:
            reasons.append('frame_count_mismatch')
        if (group != probe['source_group'] or video_id in known_ids or group in known_groups
                or media['sha256'] in known_hashes or media['sha256'] in verified_hashes
                or group in verified_groups):
            reasons.append('source_or_content_overlap')
        if not questions[video_id]:
            reasons.append('no_valid_official_item')
        record = dict(media, **info, band=band, video_id=video_id, source_group=group,
                      eligible=not reasons, reasons=reasons)
        audit.append(record)
        if not reasons:
            verified_groups.add(group)
            verified_hashes.add(media['sha256'])
            candidates[band].append(record)
    rows, labels = [], {}
    for band in BANDS:
        types = Counter()
        for record in candidates[band][:15]:
            video_id = record['video_id']
            q = min(questions[video_id], key=lambda q: (
                types[q['type'][0]], rank(video_id + ':' + q['qid'])))
            opts = [q[f'a{i}'] for i in range(5)]
            qid = f'nextqa-train-{video_id}-{q["qid"]}'
            types[q['type'][0]] += 1
            labels[qid] = opts.index(q['answer'])
            rows.append(dict(id=qid, dataset='NExT-QA', source_id=video_id,
                source_group=record['source_group'], type=q['type'], question=q['question'], options=opts,
                media_path=record['media_path'], media_sha256=record['sha256'],
                query_policy='full_clip_EOF', length_bin=band,
                duration_s=record['duration_s'], query_time_s=record['duration_s']))
    assert len(rows) == len(labels) == len({q['source_group'] for q in rows})
    assert {q['source_id'] for q in rows}.isdisjoint(known_ids)
    assert {q['media_sha256'] for q in rows}.isdisjoint(known_hashes)
    save(OUT / 'short_media_audit.json', audit)
    save(OUT / 'short_validation_questions.json', rows)
    save(OUT / 'short_validation_labels_LOCAL_ONLY.json', labels)
    return dict(questions=len(rows), bins=dict(Counter(q['length_bin'] for q in rows)),
                candidate_bins={band: len(candidates[band]) for band in BANDS},
                verified_source_scope='VidOR mapped source and exact content SHA; official NExT-QA train annotations',
                known_video_ids=len(known_ids), known_source_groups=len(known_groups),
                known_media_sha256=len(known_hashes),
                rejections=dict(Counter(reason for a in audit for reason in a['reasons'])))


def build_stream(complete):
    main = read(PRIOR / 'new_streamingbench_candidates.json')
    diagnosis = read(PRIOR / 'new_diagnosis_source_queue.json')
    main_by_media = defaultdict(list)
    for q in main:
        main_by_media[q['source_id']].append(q)
    diagnosis_by_media = {q['source_id']: q for q in diagnosis}
    previous = read(ROOT / 'runs/streamingbench_official_tail_20261031_01/complete_manifest.json')
    blocked = {r['sha256'] for r in previous['files']}
    _, _, _, short_hashes = short_provenance()
    blocked.update(short_hashes)
    manifest, main_ready, diag_ready = [], [], []
    for media in complete:
        name = media['name']
        panel = media['panel']
        source = diagnosis_by_media[name] if panel == 'diagnosis' else None
        tasks = [source['early_anchor'], source['current_anchor']] if source else main_by_media[name]
        try:
            info = media_record(media, query_times=[q['query_time_s'] for q in tasks])
        except (av.FFmpegError, AssertionError, OSError, ValueError, IndexError) as error:
            manifest.append(dict(media, technical_pass=False, reasons=[type(error).__name__]))
            continue
        pass_pts = {float(query) for query, pts in info['near_cutoff_pts'].items()
                    if pts is not None and 0 <= float(query) - pts <= 1}
        reasons = []
        if media['sha256'] in blocked:
            reasons.append('content_sha_overlap')
        if not pass_pts:
            reasons.append('no_decodable_frame_near_cutoff')
        manifest.append(dict(media, **info, technical_pass=not reasons, reasons=reasons,
                             parent_capture_verified=False))
        if reasons:
            continue
        blocked.add(media['sha256'])
        fields = dict(media_path=media['media_path'], media_sha256=media['sha256'], duration_s=info['duration_s'])
        if source and all(q['query_time_s'] in pass_pts for q in tasks):
            diag_ready.append(dict(source, **fields, status='needs_visual_evidence_and_parent_review'))
        elif not source:
            main_ready.extend(dict(q, **fields, status='technical_pass_parent_capture_unresolved')
                              for q in tasks if q['query_time_s'] in pass_pts)
    save(OUT / 'stream_media_audit.json', manifest)
    save(OUT / 'stream_main_technical_candidates.json', main_ready)
    save(OUT / 'diagnosis_evidence_review_queue.json', diag_ready)
    return dict(main_sources=len({q['source_id'] for q in main_ready}), main_questions=len(main_ready),
                diagnosis_sources=len(diag_ready), formal_admissions=0,
                unresolved='original capture identities and evidence review for diagnosis',
                rejected_media={r['name']: r['reasons'] for r in manifest if r['reasons']})


if __name__ == '__main__':
    result = read(OUT / 'acquisition.json')
    assert result['status'] in {'complete', 'partial', 'stopped'}
    complete = result['complete']
    short = build_short([r for r in complete if r['panel'] == 'short'])
    stream = build_stream([r for r in complete if r['panel'] != 'short'])
    summary = dict(short=short, streamingbench=stream, verified_media=len(complete),
                   model_inference=False)
    save(OUT / 'audit_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))
