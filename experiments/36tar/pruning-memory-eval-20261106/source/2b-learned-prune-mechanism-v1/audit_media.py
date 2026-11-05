"""CPU, single-thread full decode and deterministic sampled-input identity audit."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path, PurePosixPath
import av
from PIL import Image, ImageOps
from acquire_media import verify


def canvas(frame):
    image = frame.to_image().convert('RGB')
    image = ImageOps.contain(image, (384, 288), method=Image.Resampling.BICUBIC)
    output = Image.new('RGB', (384, 288))
    output.paste(image, ((384-image.width)//2, (288-image.height)//2))
    return output


def audit(path, row):
    assert verify(path, row), 'media identity mismatch'
    samples = []; count = 0; first = None; last = None; end = None; previous = None; gaps = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.thread_count = 1
        fps = float(stream.average_rate) if stream.average_rate else None
        metadata_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        next_target = 0.0
        for frame in container.decode(stream):
            assert frame.pts is not None, 'missing video PTS'
            t = float(frame.pts * stream.time_base)
            if first is None:
                first = t
            t -= first
            assert previous is None or t > previous, 'non-increasing PTS'
            if previous is not None:
                gaps.append(t-previous)
            previous = t; last = t; count += 1
            duration = float(frame.duration * stream.time_base) if frame.duration else (1/fps if fps else 0)
            end = t + duration
            if t + 1e-9 >= next_target:
                image = canvas(frame)
                samples.append(dict(frame_index=count-1, pts_s=t, target_s=next_target,
                                    rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                # No duplicating a decoded frame to fill missing sampled instants.
                next_target = (int((t+1e-9)*2)+1)/2
    reasons = []
    duration_s = max(v for v in [end, metadata_duration] if v is not None)
    if duration_s > 20 + 1e-6: reasons.append('duration_exceeds_20s')
    if not 2 <= len(samples) <= 40: reasons.append('sample_count_outside_2_to_40')
    if count != row['frame_count']: reasons.append('official_frame_count_mismatch')
    if max(gaps, default=0) > 0.5 + 1e-6: reasons.append('PTS_gap_exceeds_sampling_interval')
    return dict(video_id=row['video_id'], role=row['role'], duration_s=duration_s, decoded_frames=count,
                metadata_duration_s=metadata_duration, actual_fps=fps, last_pts_s=last,
                samples=samples, real_frames=len(samples), encoding_pairs=(len(samples)+1)//2,
                duplicate_tail_for_encoding=bool(len(samples)%2), reasons=reasons,
                admitted=not reasons, sampled_answer_support='not_independently_reviewed')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--media-root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); plan = json.loads(args.plan.read_text(encoding='utf-8'))
    started = time.monotonic(); rows = []
    for s in plan['files']:
        rel = PurePosixPath(s['mirror_path']); assert not rel.is_absolute() and '..' not in rel.parts
        try:
            row = audit(args.media_root / Path(*rel.parts), s)
        except Exception as exc:
            row = dict(video_id=s['video_id'], role=s['role'], admitted=False,
                       reasons=['decode_or_identity_failure'], error=type(exc).__name__ + ': ' + str(exc))
        rows.append(row)
        if len(rows)%20 == 0: print(json.dumps(dict(audited=len(rows), seconds=time.monotonic()-started)), flush=True)
    admitted = [r for r in rows if r['admitted']]
    counts = collections.Counter(r['role'] for r in admitted)
    gate = counts['train'] >= 100 and counts['development'] >= 20 and counts['confirmation'] >= 20
    result = dict(status='admission_gate_passed' if gate else 'admission_gate_failed_stop',
                  admitted=len(admitted), admitted_roles=dict(counts), audited=len(rows),
                  exclusions=dict(collections.Counter(x for r in rows for x in r['reasons'])),
                  seconds=time.monotonic()-started, rows=rows, GPU=0,
                  answer_support_audit=False, plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    main()
