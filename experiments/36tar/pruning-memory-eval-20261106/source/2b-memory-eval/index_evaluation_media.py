"""Single-thread actual-frame indexing; no pretrained models or QA outputs.

0.5FPS for every fixed non-development asset; additionally 1FPS for GQA comparison
annotation only. Holdout receives no contact sheets or policy performance audit.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import av
from PIL import Image,ImageStat

from audit_sampling import canvas_bytes


def fingerprint(rgb):
    image=Image.frombytes('RGB',(256,192),rgb).convert('L')
    pixels=list(image.resize((9,8),Image.Resampling.BILINEAR).getdata())
    value=0
    for y in range(8):
        for x in range(8):value=(value<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
    return dict(dhash64=f'{value:016x}',gray_std=ImageStat.Stat(image).stddev[0])


def index(path,max_query,extra_1fps):
    start=time.monotonic();rates=[.5,1.] if extra_1fps else [.5]
    due={rate:0. for rate in rates};frames={rate:[] for rate in rates}
    origin=None;previous=None;count=0;boundary_seen=False
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        metadata=dict(width=stream.width,height=stream.height,
            average_fps=float(stream.average_rate) if stream.average_rate else None,
            container_duration_s=float(container.duration/av.time_base) if container.duration else None,
            video_duration_s=float(stream.duration*stream.time_base) if stream.duration else None)
        for frame_id,frame in enumerate(container.decode(stream)):
            count+=1
            if time.monotonic()-start>120:raise TimeoutError('per-source indexing budget')
            if frame.time is None:raise ValueError('missing real timestamp')
            if origin is None:origin=float(frame.time)
            pts=float(frame.time)-origin
            if previous is not None and pts<=previous:raise ValueError('non-increasing real PTS')
            previous=pts
            if pts>max_query+1e-9:
                boundary_seen=True;break
            selected=[rate for rate in rates if pts+1e-9>=due[rate]]
            if not selected:continue
            rgb=canvas_bytes(frame)
            item=dict(frame_id=frame_id,pts_s=pts,rgb_sha256=hashlib.sha256(rgb).hexdigest(),**fingerprint(rgb))
            for rate in selected:
                frames[rate].append(item)
                due[rate]=(math.floor(pts*rate+1e-9)+1)/rate
        if not all(frames.values()):raise ValueError('empty real prefix')
    duration=metadata['video_duration_s'] or metadata['container_duration_s']
    return dict(**metadata,first_pts_s=origin,max_query_s=max_query,
        declared_query_minus_video_duration_s=max_query-duration if duration else None,
        eof_before_query=not boundary_seen and previous<max_query,
        decoded_frames=count,last_decoded_pts_s=previous,elapsed_s=time.monotonic()-start,
        frames_05fps=frames[.5],frames_1fps=frames.get(1.))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=('nextgqa','ovo'),required=True)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    newdir='nextgqa-eval48-hfverified-20261018' if args.dataset=='nextgqa' else 'ovo-eval32-20261018'
    download=root/'datasets/2b-memory-eval'/newdir/'download_manifest.json'
    media={f['source_path']:f for f in json.loads(download.read_text(encoding='utf-8'))['files']}
    sources=list(dict.fromkeys(r['source_path'] for r in rows if r['dataset']==args.dataset and r['split']!='development'))
    if any(s not in media for s in sources):raise ValueError('all planned assets must be acquired before indexing')
    if args.out.exists():raise FileExistsError('new evidence output required')
    report=dict(status='actual_cpu_media_index_no_visual_review_or_model_accuracy',
        dataset=args.dataset,source_manifest_sha256=hashlib.sha256(download.read_bytes()).hexdigest(),
        canvas_hw=[192,256],main_sampling_fps=.5,annotation_sampling_fps=1.,
        videos=[],failures=[])
    for source in sources:
        qs=[r for r in rows if r['dataset']==args.dataset and r['source_path']==source]
        item=media[source];path=Path(item['local_path'])
        if path.stat().st_size!=item['bytes']:raise ValueError('downloaded size changed')
        try:
            trace=index(path,max(q['query_time_s'] for q in qs),
                args.dataset=='nextgqa' and qs[0]['split']=='comparison')
            report['videos'].append(dict(source_path=source,split=qs[0]['split'],
                question_ids=[q['id'] for q in qs],source_bytes=item['bytes'],source_sha256=item['sha256'],
                local_path=str(path),**trace))
            print(json.dumps(dict(source=source,split=qs[0]['split'],sampled=len(trace['frames_05fps']),
                seconds=round(trace['elapsed_s'],2))),flush=True)
        except Exception as error:
            report['failures'].append(dict(source_path=source,error_type=type(error).__name__))
            print(json.dumps(dict(source=source,error_type=type(error).__name__)),flush=True)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed=len(report['videos']),failures=len(report['failures']),
        sampled_05fps=sum(len(v['frames_05fps']) for v in report['videos']),
        seconds=sum(v['elapsed_s'] for v in report['videos']))))


if __name__=='__main__':main()
