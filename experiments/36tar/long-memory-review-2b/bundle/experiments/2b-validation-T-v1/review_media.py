"""Prepare every causal1FPS canvas for blind content review, with bounded RAM.

Creates evidence images and identity screens; never assigns content verdicts.
Only one24-frame page is buffered. No pretrained model or GPU is imported.
"""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import sys
import time

import av
from PIL import Image, ImageDraw, ImageFont
import psutil

from prepare import OUT, ROOT, read, save, sha

sys.path.insert(0, str(ROOT/'experiments/2b-memory-eval'))
from audit_sampling import canvas_bytes, query_snapshot
from index_evaluation_media import fingerprint
from build_dataset_release import load_media_traces


def freeze():
    frozen = read(OUT/'metadata_freeze.json')
    assert sha(OUT/'candidate_questions119.json') == frozen['output_sha256']['candidate_questions119.json']
    paths = [Path(__file__), ROOT/'experiments/2b-memory-eval/audit_sampling.py',
             ROOT/'experiments/2b-memory-eval/index_evaluation_media.py',
             ROOT/'experiments/2b-memory-eval/build_dataset_release.py']
    save(OUT/'review_plan.json', dict(
        status='prospective_pixel_review_protocol_no_model_outputs',
        candidate_sha256=sha(OUT/'candidate_questions119.json'),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        fps=1, canvas_hw=[192,256], resize='existing matched two-step letterbox',
        full_prefix_all_samples=True, question_interval_does_not_select_frames=True,
        frames_per_page=24, maximum_buffered_RGB_frames=24,
        per_source_seconds=120, total_seconds=900, process_RSS_cap=2**30,
        PNG_byte_cap=256*2**20, fingerprint_fractions=[0,.25,.5,.75,1],
        perceptual_screen=dict(max_hamming=4, minimum_gray_std=8, minimum_distinct_frames=2),
        blind_to_all_model_predictions=True, content_verdicts_after_actual_viewing_only=True,
        preserve_ambiguities=True, no_label_rewriting=True,
        no_claim_parent_independence_from_hash_screen=True,
        single_assistant_review=True, independent_second_review='not_performed',
        old_necessity='not_proven_by_intervals_or_visibility', GPU_jobs=0, cluster_operations=0))
    print('Prospective pixel review plan frozen; no media decoded.')


def run(recovery=False):
    plan = read(OUT/'review_plan.json')
    for path, digest in plan['source_sha256'].items():
        if recovery and path == Path(__file__).relative_to(ROOT).as_posix():
            assert sha(OUT/'review_media_initial_snapshot.py') == digest
        else:
            assert sha(ROOT/path) == digest, path
    assert sha(OUT/'candidate_questions119.json') == plan['candidate_sha256']
    manifest = read(OUT/'download_manifest.json')
    assert manifest['status'] == 'all40_integrity_verified_NOT_content_admitted'
    roster = read(OUT/'candidate_questions119.json')
    folder = OUT/'sheets'
    folder.mkdir(exist_ok=recovery)
    if recovery:
        assert read(OUT/'render_failure_diagnostic.json')['failed_source'] == '1015/13884293626'
        save(OUT/'render_recovery_started.json',dict(
            reason='source40 hit120s wall guard; codec header and300-frame bounded decode probe succeeded',
            original_source_sha256=sha(OUT/'review_media_initial_snapshot.py'),
            repaired_source_sha256=sha(__file__),maximum_recovery_attempts=1,
            last_source_wall_seconds=300,total_wall_seconds=1200,
            sampling_canvas_and_question_selection_unchanged=True,
            existing_images_reused_only_after_actual_RGB_crop_checks=True))
    started = time.monotonic()
    font = ImageFont.load_default()
    traces, questions, images = [], [], []
    png_bytes = 0
    crop_checks = 0
    peak_rss = 0
    for source in manifest['files']:
        path = ROOT/source['media_path']
        assert path.resolve().is_relative_to((ROOT/'datasets/2b-memory-eval/T-new40-20261020').resolve())
        assert path.stat().st_size == source['bytes'] and sha(path) == source['sha256']
        qs = [q for q in roster if q['source_path'] == source['source_path']]
        assert len(qs) in (2,3)
        query = max(q['query_time_s'] for q in qs)
        frames, buffer, pages = [], [], []
        origin, previous, due = None, None, 0.
        decoded = 0
        source_start = time.monotonic()

        def flush():
            nonlocal png_bytes, crop_checks
            rows = math.ceil(len(buffer)/4)
            sheet = Image.new('RGB', (1024,rows*214), 'white')
            draw = ImageDraw.Draw(sheet)
            locations = []
            for k, (spec, rgb) in enumerate(buffer):
                x, y = (k%4)*256, (k//4)*214
                draw.text((x+3,y+3), f"{spec['pts_s']:.2f}s frame{spec['frame_id']}", fill='black', font=font)
                sheet.paste(Image.frombytes('RGB', (256,192), rgb), (x,y+22))
                locations.append(dict(frame_id=spec['frame_id'],pts_s=spec['pts_s'],
                    rgb_sha256=spec['rgb_sha256'],crop_xyxy=[x,y+22,x+256,y+214]))
            image_path = folder/f"{source['source_id']}-p{len(pages)+1:02d}.png"
            if not image_path.exists():
                sheet.save(image_path)
            else:
                assert recovery, 'original attempt cannot overwrite PNG'
            with Image.open(image_path) as reopened:
                for item in locations:
                    assert hashlib.sha256(reopened.crop(item['crop_xyxy']).tobytes()).hexdigest() == item['rgb_sha256']
                    crop_checks += 1
            png_bytes += image_path.stat().st_size
            assert png_bytes <= plan['PNG_byte_cap'], 'PNG disk budget'
            page = dict(path=image_path.relative_to(ROOT).as_posix(),sha256=sha(image_path),frames=locations)
            pages.append(page)
            images.append(page)
            buffer.clear()
            sheet.close()

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            metadata = dict(width=stream.width,height=stream.height,
                average_fps=float(stream.average_rate) if stream.average_rate else None,
                container_duration_s=float(container.duration/av.time_base) if container.duration else None,
                video_duration_s=float(stream.duration*stream.time_base) if stream.duration else None)
            for index, frame in enumerate(container.decode(stream)):
                source_limit = 300 if recovery and source['source_id']=='13884293626' else plan['per_source_seconds']
                assert time.monotonic()-source_start < source_limit, 'source decode time budget'
                assert time.monotonic()-started < (1200 if recovery else plan['total_seconds']), 'total decode time budget'
                decoded += 1
                assert frame.time is not None
                if origin is None:
                    origin = float(frame.time)
                pts = float(frame.time)-origin
                assert previous is None or pts > previous
                previous = pts
                if pts > query+1e-9:
                    break
                if pts+1e-9 < due:
                    continue
                rgb = canvas_bytes(frame)
                item = dict(frame_id=index,pts_s=pts,rgb_sha256=hashlib.sha256(rgb).hexdigest(),**fingerprint(rgb))
                frames.append(item)
                buffer.append((item,rgb))
                due = math.floor(pts+1e-9)+1.
                peak_rss = max(peak_rss,psutil.Process().memory_info().rss)
                assert peak_rss < plan['process_RSS_cap'], 'CPU process RSS budget'
                if len(buffer) == 24:
                    flush()
        if buffer:
            flush()
        assert frames and len(frames) == sum(len(p['frames']) for p in pages)
        traces.append(dict(source_path=source['source_path'],source_group_id=source['source_group_id'],
            media_path=source['media_path'],media_sha256=source['sha256'],fps=1,
            max_query_s=query,pts_origin_s=origin,last_decoded_pts_s=previous,
            decoded_frames=decoded,frames=frames,pages=pages,
            preparation_seconds=time.monotonic()-source_start,**metadata))
        for q in qs:
            questions.append(dict(id=q['id'],source_path=q['source_path'],selection_role=q['selection_role'],
                snapshot=query_snapshot(frames,q['query_time_s'],q['evidence_intervals']),
                content_verdict='pending_actual_page_review',old_evidence_necessity_proven=False))
        print(dict(prepared_sources=len(traces),frames=len(frames),pages=len(pages)),flush=True)
    save(OUT/'sampling_all40_1fps.json', dict(videos=traces,questions=questions))
    save(OUT/'image_receipts.json', images)

    # Reuse immutable old identities; this is a weak overlap screen, not proof
    # that two distinct IDs originate from independent filming sessions.
    _, old_traces = load_media_traces()
    old_new24 = read(ROOT/'runs/memory_target_new24_20261019_01/sampling_all24_05fps.json')['videos']
    old = list(old_traces.values()) + old_new24
    assert len(old) == 124
    old_assets = {x['sha256']:x['original_source_path'] for x in read(
        ROOT/'runs/2b_collaboration_sync_20261020_01/media_manifest124.json')}
    old_frames = defaultdict(set)
    for t in old:
        for frame in t['frames']:
            old_frames[frame['rgb_sha256']].add(t['source_path'])

    def sketch(t):
        fs = t['frames']
        return dict(source_path=t['source_path'],frames=[fs[i] for i in sorted(
            {round((len(fs)-1)*f) for f in [0,.25,.5,.75,1]})])

    old_sketches = read(ROOT/'runs/dataset_build_20261018_01/source_overlap_screen.json')['sketches']
    old_sketches += [sketch(t) for t in old_new24]
    new_sketches = [sketch(t) for t in traces]
    exact_frames, weak, exact_assets = [], [], []
    for t in traces:
        if t['media_sha256'] in old_assets:
            exact_assets.append(dict(new=t['source_path'],old=old_assets[t['media_sha256']]))
        for frame in t['frames']:
            if frame['rgb_sha256'] in old_frames:
                exact_frames.append(dict(new=t['source_path'],pts_s=frame['pts_s'],
                    old=sorted(old_frames[frame['rgb_sha256']])))
    for i,a in enumerate(new_sketches):
        for b in old_sketches+new_sketches[:i]:
            matches = []
            for x in a['frames']:
                for y in b['frames']:
                    if x['gray_std'] < 8 or y['gray_std'] < 8:
                        continue
                    distance = (int(x['dhash64'],16)^int(y['dhash64'],16)).bit_count()
                    if distance <= 4:
                        matches.append(dict(a_frame=x['frame_id'],b_frame=y['frame_id'],hamming=distance))
            if len({m['a_frame'] for m in matches}) >= 2 and len({m['b_frame'] for m in matches}) >= 2:
                weak.append(dict(a=a['source_path'],b=b['source_path'],matches=matches,
                    verdict='pending_visual_parent_review'))
    save(OUT/'overlap_screen.json', dict(old_sources=124,new_sources=40,
        exact_old_assets=exact_assets,exact_old_sampled_frames=exact_frames,
        weak_perceptual_candidates=weak,automatic_exclusion=False,
        reused_old_receipts=True,filming_independence_proven=False))
    receipt = dict(status='complete_prefix_images_ready_content_review_PENDING',
        sources=40,questions=119,frames=crop_checks,pages=len(images),PNG_bytes=png_bytes,
        peak_process_RSS_bytes=peak_rss,seconds=time.monotonic()-started,
        all_crops_match_actual_RGB=True,maximum_buffered_RGB_frames=24,
        exact_old_assets=len(exact_assets),exact_old_sampled_frames=len(exact_frames),
        weak_perceptual_candidates=len(weak),content_reviews_completed=0,
        models_loaded=0,GPU_jobs=0,cluster_operations=0,
        source_sha256=sha(__file__),review_plan_sha256=sha(OUT/'review_plan.json'))
    if recovery:
        for item in read(OUT/'pre_recovery_image_manifest.json'):
            assert sha(OUT/item['path']) == item['sha256'], 'prior PNG changed'
        receipt.update(engineering_recovery_attempts=1,prior_PNGs_unchanged=True,
                       initial_failure_preserved=True)
    save(OUT/'render_receipt.json', receipt)
    print(receipt,flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'recover-once':
        run(recovery=True)
    else:
        {'freeze':freeze,'run':run}[sys.argv[1]]()
