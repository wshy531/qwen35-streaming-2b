"""Bounded local decode: causal sampled-frame identities, not model evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import av
from PIL import Image

from memory_policy import Pair, RecentFIFO, RecentLongHierarchy


def contains_all(intervals, timestamps):
    if not intervals:
        return None
    return all(any(a-1e-9 <= t <= b+1e-9 for t in timestamps) for a,b in intervals)


def canvas_bytes(frame):
    image = frame.to_image()
    # Same two-step resize as first_2b_baseline.frames_from, no torch needed.
    image.thumbnail((256,256), Image.Resampling.BICUBIC)
    scale = min(256/image.width, 192/image.height)
    size = (max(1,min(256,round(image.width*scale))),
            max(1,min(192,round(image.height*scale))))
    image = image.resize(size, Image.Resampling.BICUBIC)
    canvas = Image.new('RGB',(256,192),(0,0,0))
    canvas.paste(image,((256-size[0])//2,(192-size[1])//2))
    return canvas.tobytes()


def sample(path, max_query, fps, seconds=120):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('finite positive fps required')
    result, due, origin, previous = [], 0., None, None
    start = time.monotonic()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if time.monotonic()-start > seconds:
                raise TimeoutError('bounded single-video decode exceeded')
            if frame.time is None:
                raise ValueError('missing timestamp')
            if origin is None:
                origin = float(frame.time)
            pts = float(frame.time)-origin
            if previous is not None and pts <= previous:
                raise ValueError('non-increasing real PTS')
            previous = pts
            if pts > max_query+1e-9:
                break  # decoded future frame is never transformed/recorded as input
            if pts+1e-9 < due:
                continue
            result.append(dict(frame_id=index, pts_s=pts,
                rgb_sha256=hashlib.sha256(canvas_bytes(frame)).hexdigest()))
            due = (math.floor(pts*fps+1e-9)+1)/fps
    return dict(frames=result, elapsed_s=time.monotonic()-start, canvas_hw=[192,256],
                fps=fps, pts_origin_s=origin, decoded_until_s=previous)


def query_snapshot(frames, query, intervals, provisional=None):
    prefix = [f for f in frames if f['pts_s'] <= query+1e-9]
    m0, m1 = RecentFIFO(8), RecentLongHierarchy(4,4)
    pairs = []
    for i in range(0,len(prefix),2):
        a, b = prefix[i], prefix[min(i+1,len(prefix)-1)]
        pairs.append(dict(index=i//2, frame_ids=[a['frame_id'],b['frame_id']],
            pts_s=[a['pts_s'],b['pts_s']], padded_tail=i+1==len(prefix)))
        pair = Pair(i//2,a['pts_s'],b['pts_s'],pairs[-1])
        m0.append(pair)
        m1.append(pair)
    def view(policy):
        kept = policy.read()
        times = [t for p in kept for t in p.payload['pts_s']]
        return dict(pair_indices=[p.index for p in kept], real_frame_ids=sorted(
            {v for p in kept for v in p.payload['frame_ids']}),
            all_verified_intervals_hit=contains_all(intervals,times),
            provisional_window_hit=contains_all([provisional],times) if provisional else None)
    return dict(query_time_s=query, sampled_real_frames=len(prefix), pairs=len(pairs),
        odd_tail_padding=bool(len(prefix)%2),
        padding_policy='query_snapshot_only_last_real_frame_duplicate_not_committed_to_stream',
        full_history_all_verified_intervals_hit=contains_all(intervals,[f['pts_s'] for f in prefix]),
        verified_interval_sample_hits=[dict(interval=span,frame_ids=[f['frame_id'] for f in prefix
            if span[0] <= f['pts_s'] <= span[1]]) for span in intervals or []],
        M0=view(m0), M1=view(m1), M1_provenance=m1.provenance())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--media-root',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--fps',type=float,default=.5)
    p.add_argument('--nextgqa-mp4-layout',action='store_true',
        help='Explicit author NExTVideo layout: mapped source path plus .mp4')
    args = p.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    review = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    annotations = {r['id']:r for r in review['questions']}
    sources = sorted({r['source_path'] for r in rows if r['id'] in annotations})
    out = dict(status='cpu_sampling_and_storage_coverage_only_not_accuracy',
        protocol='draft_development_only_not_frozen_benchmark',videos=[],questions=[])
    for source in sources:
        path = Path(args.media_root)/source
        if args.nextgqa_mp4_layout:
            questions_for_source=[r for r in rows if r['source_path']==source and r['id'] in annotations]
            if any(r['dataset']!='nextgqa' for r in questions_for_source):
                raise ValueError('NExT-GQA layout flag cannot alter another dataset path')
            path=Path(str(path)+'.mp4')
        if not path.resolve().is_relative_to(Path(args.media_root).resolve()):
            raise ValueError('source path escapes explicit media root')
        if not path.is_file():
            continue
        questions = [r for r in rows if r['source_path']==source and r['id'] in annotations]
        trace = sample(path,max(r['query_time_s'] for r in questions),args.fps)
        trace.update(source_path=source,source_bytes=path.stat().st_size)
        out['videos'].append(trace)
        for row in questions:
            a = annotations[row['id']]
            intervals=a.get('verified_necessary_evidence_intervals') or a.get('author_evidence_intervals')
            snapshot = query_snapshot(trace['frames'],row['query_time_s'],
                intervals,a.get('provisional_inspection_window_s'))
            snapshot.update(id=row['id'], annotation_readiness=a['readiness'],
                interval_basis='author_related_segments_not_human_necessity_verification'
                    if a.get('author_evidence_intervals') else 'review_supplied_or_unknown')
            out['questions'].append(snapshot)
        print(json.dumps(dict(source=source,sampled=len(trace['frames']),seconds=round(trace['elapsed_s'],2))),flush=True)
    Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
