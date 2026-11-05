"""Single-thread media audit; inventory is NOT question/evidence admission."""
import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import av
from PIL import Image, ImageOps
from acquire_media import verify


def audit(path, row):
    assert verify(path, row), 'identity failure'
    count = 0
    first = previous = last = None
    samples = []
    target = 0.0
    largest_gap = 0.0
    dims = set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        fps = float(stream.average_rate) if stream.average_rate else None
        declared = float(stream.duration * stream.time_base) if stream.duration is not None else None
        for frame in container.decode(stream):
            assert frame.pts is not None, 'missing PTS'
            raw = float(frame.pts * stream.time_base)
            if first is None:
                first = raw
            t = raw - first
            if previous is not None:
                assert t > previous, 'PTS not increasing'
                largest_gap = max(largest_gap, t-previous)
            previous = last = t
            dims.add((frame.width, frame.height))
            count += 1
            if t + 1e-9 >= target:
                rgb = ImageOps.contain(frame.to_image().convert('RGB'), (384, 288), Image.Resampling.BICUBIC)
                canvas = Image.new('RGB', (384, 288))
                canvas.paste(rgb, ((384-rgb.width)//2, (288-rgb.height)//2))
                samples.append(dict(frame_index=count-1, pts_s=t, raw_pts=frame.pts,
                                    time_base=str(stream.time_base), target_s=target,
                                    rgb_sha256=hashlib.sha256(canvas.tobytes()).hexdigest()))
                target = (int((t+1e-9)*2)+1)/2
        assert count > 0
    duration = max(last + (1/fps if fps else 0), declared or 0)
    flags = []
    if count != row['frame_count']:
        flags.append('official_frame_count_mismatch')
    if largest_gap > .5 + 1e-6:
        flags.append('PTS_gap_exceeds_sampling_interval')
    return dict(video_id=row['video_id'], role=row['role'], duration_s=duration,
                frame_count=count, actual_fps=fps, dimensions=sorted(dims),
                first_pts_s=first, last_pts_s=last, max_gap_s=largest_gap,
                samples=samples, complete_real_pairs=len(samples)//2,
                unpaired_tail=bool(len(samples)%2), flags=flags,
                media_check_passed=not flags, delayed_QA_admitted=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--media-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wait-for-download', action='store_true')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    assert len(plan['files']) == 150
    started = time.monotonic()
    rows = []
    with args.out.open('x', encoding='utf-8') as out:
        for row in plan['files']:
            path = args.media_root / row['mirror_path']
            while not path.exists() and args.wait_for_download:
                receipt = args.plan.parent / 'media_acquisition.json'
                if receipt.exists() or time.monotonic()-started > 45*60:
                    break
                time.sleep(2)
            if not path.exists():
                break
            try:
                result = audit(path, row)
            except Exception as exc:
                result = dict(video_id=row['video_id'], role=row['role'],
                              media_check_passed=False, delayed_QA_admitted=False,
                              flags=['media_audit_error'], error=type(exc).__name__)
            rows.append(result)
            out.write(json.dumps(result)+'\n')
            out.flush()
            if len(rows)%10 == 0:
                print(json.dumps(dict(audited=len(rows), seconds=time.monotonic()-started)), flush=True)
    durations = [r['duration_s'] for r in rows if 'duration_s' in r]
    histogram = collections.Counter()
    for t in durations:
        histogram['<=30s' if t<=30 else '30-60s' if t<=60 else '60-120s' if t<=120 else '>120s'] += 1
    summary = dict(status='inventory_complete' if len(rows)==150 else 'inventory_partial',
                   audited=len(rows), media_passed=sum(r['media_check_passed'] for r in rows),
                   delayed_QA_admitted=0, duration_histogram=dict(histogram),
                   total_duration_s=sum(durations), min_duration_s=min(durations, default=None),
                   max_duration_s=max(durations, default=None),
                   roles=dict(collections.Counter(r['role'] for r in rows)),
                   flags=dict(collections.Counter(f for r in rows for f in r['flags'])),
                   seconds=time.monotonic()-started, GPU=0,
                   plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    with args.out.with_suffix('.summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
