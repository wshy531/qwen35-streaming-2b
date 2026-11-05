"""Single-thread CPU decode/hash audit; never imports torch or touches partner files."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time
from common import ROOT, INPUTS, PREP, read, save, sha


def canvas(frame):
    from PIL import Image
    image = frame.to_image()
    image.thumbnail((384, 384), Image.Resampling.BICUBIC)
    scale = min(384 / image.width, 288 / image.height)
    size = (max(1, min(384, round(image.width * scale))), max(1, min(288, round(image.height * scale))))
    image = image.resize(size, Image.Resampling.BICUBIC)
    result = Image.new('RGB', (384, 288))
    result.paste(image, ((384 - size[0]) // 2, (288 - size[1]) // 2))
    return result


def sheet(images, title, output):
    from PIL import Image, ImageDraw
    if not images or len(images) > 24:
        raise ValueError('bounded review sheet required')
    result = Image.new('RGB', (1536, 32 + math.ceil(len(images) / 4) * 312), 'white')
    draw = ImageDraw.Draw(result)
    draw.text((5, 5), title, fill='black')
    for i, (record, image) in enumerate(images):
        x, y = (i % 4) * 384, 32 + (i // 4) * 312
        draw.text((x + 3, y + 3), f"PTS {record['pts_s']:.3f}s / frame {record['frame_id']}", fill='black')
        result.paste(image, (x, y + 24))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    result.save(output)
    return dict(relative_path=output.relative_to(ROOT).as_posix(), sha256=sha(output), bytes=output.stat().st_size)


def audit(media_root, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CPU-only media audit requires CUDA_VISIBLE_DEVICES=-1')
    import av
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    labels = {r['question_id']: r for r in read(ROOT / 'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json')}
    rows = []
    started = time.monotonic()
    for pair in manifest['pairs']:
        path = media_root / Path(pair['source_path']).name
        if path.resolve().parent != media_root.resolve():
            raise ValueError('explicit single media directory required')
        if path.stat().st_size != pair['media_bytes'] or sha(path) != pair['media_sha256']:
            raise ValueError('media size/SHA mismatch')
        label = labels[pair['old_question_id'][7:]]
        begin = 2 * math.floor(label['answer_start_time'] / 2)
        end = 2 * math.ceil((label['answer_end_time'] + 1) / 2) - 1
        if not 0 <= begin <= end <= pair['committed_through_s'] or end - begin >= 24:
            raise ValueError('bounded author evidence span required; do not silently subsample')
        recent_ids = {r['frame_id']: r for r in pair['recent_frames']}
        donor_images, old_images, recent_images = [], [], []
        trace = []
        origin, due = None, 0.0
        previous = -1.0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for frame_id, frame in enumerate(container.decode(stream)):
                if time.monotonic() - started > 110:
                    raise TimeoutError('bounded login-node CPU audit exceeded 110 seconds')
                if frame.time is None:
                    raise ValueError('display timestamp required')
                if origin is None:
                    origin = float(frame.time)
                pts = float(frame.time) - origin
                if not math.isfinite(pts) or pts <= previous:
                    raise ValueError('strict display PTS required')
                previous = pts
                if pts > pair['committed_through_s'] + 1e-9:
                    break
                if pts + 1e-9 < due:
                    continue
                image = canvas(frame)
                record = dict(frame_id=frame_id, pts_s=pts,
                              mid_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                trace.append(record)
                if len(trace) <= 8:
                    donor_images.append((record, image))
                if begin <= pts <= end:
                    old_images.append((record, image))
                if frame_id in recent_ids:
                    if record != recent_ids[frame_id]:
                        raise ValueError('partner recent sampling/resize SHA mismatch')
                    recent_images.append((record, image))
                due = math.floor(pts + 1e-9) + 1
        if (len(trace) != 2 * pair['committed_pairs'] or len(trace) % 2
                or trace[-8:] != pair['recent_frames'] or len(recent_images) != 8
                or len(donor_images) != 8 or donor_images[-1][0]['pts_s'] > 7.0):
            raise ValueError('causal prefix/recent/donor trace mismatch')
        cards = PREP / 'review_cards_LOCAL_ONLY'
        old_card = sheet(old_images, pair['pair_id'] + ' / author evidence interval (not a necessity proof)',
                         cards / (pair['pair_id'] + '_old.png'))
        donor_card = sheet(donor_images, pair['pair_id'] + ' / first 4 complete pairs; donor source remains explicit',
                           cards / (pair['pair_id'] + '_donor.png'))
        rows.append(dict(pair_id=pair['pair_id'], media_name=path.name, media_sha256=pair['media_sha256'],
            sampled_frames=len(trace), complete_pairs=len(trace) // 2,
            committed_through_s=trace[-1]['pts_s'], recent_exact_partner_RGB_and_PTS=True,
            sampled_trace=trace, donor_trace=trace[:8], author_span_s=[label['answer_start_time'], label['answer_end_time']],
            old_review_trace=[r for r, _ in old_images], old_card=old_card, donor_card=donor_card,
            old_span_is_official_localization_hint_not_independent_validation=True))
    save(output, dict(status='passed_CPU_media_only', sources=rows,
        complete_target_pairs=sum(r['complete_pairs'] for r in rows),
        latest_frames_are_exact_partner_committed_input=True, decoder_threads=1,
        bounded_RGB_review_buffer_frames_max=40, GPU_jobs=0, model_loads=0,
        partner_writes=0, independent_review='pending',
        wall_seconds=time.monotonic() - started, manifest_sha256=sha(INPUTS / 'manifest_LOCAL_ONLY.json')))
    print(dict(status='passed_CPU_media_only', sources=len(rows), pairs=sum(r['complete_pairs'] for r in rows),
               partner_writes=0, GPU_jobs=0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--media-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.media_root, args.output)
