"""Bounded overview of14 existing long media, no annotation/model inference."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import av
from PIL import Image,ImageDraw
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/2b_spatial_time_prepare_20261021_01'
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from audit_sampling import canvas_bytes


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    old=ROOT/'runs/2b_long_stream_prepare_20261020_01'
    used={s['source_path'] for s in read(old/'stream_sources6.json')}
    sources=[s for s in read(old/'local_long_video_inventory.json')['media'] if s['duration_s']>=300 and s['source_path'] not in used]
    assert len(sources)==14
    save(OUT/'overview_plan14.json',dict(sources=14,sampling_seconds=30,RSS_cap_bytes=2**30,PNG_cap_bytes=96*2**20,
        total_seconds=1800,maximum_buffered_frames=24,scope='overview qualification only; no question labels or continuity proof',GPU_jobs=0))
    folder=OUT/'overview14';folder.mkdir(exist_ok=False);start=time.monotonic();total=peak=0;videos=[]
    for number,s in enumerate(sources):
        path=Path(s['local_path']);assert sha(path)==s['declared_sha256'];frames=[];origin=previous=None;due=0.;decoded=0
        began=time.monotonic()
        with av.open(str(path)) as c:
            stream=c.streams.video[0];stream.codec_context.thread_count=1
            for i,frame in enumerate(c.decode(stream)):
                assert time.monotonic()-start<1800 and time.monotonic()-began<240
                assert frame.time is not None
                if origin is None:origin=float(frame.time)
                pts=float(frame.time)-origin;assert previous is None or pts>previous;previous=pts;decoded+=1
                if pts+1e-9<due:continue
                rgb=canvas_bytes(frame);meta=dict(frame_id=i,pts_s=pts,rgb_sha256=hashlib.sha256(rgb).hexdigest())
                frames.append((meta,rgb));assert len(frames)<=24
                due=(math.floor(pts/30+1e-9)+1)*30
                peak=max(peak,psutil.Process().memory_info().rss);assert peak<2**30
        assert sha(path)==s['declared_sha256'] and frames
        sheet=Image.new('RGB',(1024,math.ceil(len(frames)/4)*214),'white');draw=ImageDraw.Draw(sheet);metadata=[]
        for j,(meta,rgb) in enumerate(frames):
            x,y=j%4*256,j//4*214;draw.text((x+2,y+3),f'{number:02d} {meta["pts_s"]:.2f}s id{meta["frame_id"]}',fill='black')
            sheet.paste(Image.frombytes('RGB',(256,192),rgb),(x,y+22));metadata.append(dict(**meta,crop=[x,y+22,x+256,y+214]))
        dest=folder/f'{number:02d}-{Path(s["source_path"]).stem}.png';sheet.save(dest);sheet.close()
        total+=dest.stat().st_size;assert total<96*2**20
        with Image.open(dest) as image:
            for meta,_ in frames:
                m=next(m for m in metadata if m['frame_id']==meta['frame_id'])
                assert hashlib.sha256(image.crop(m['crop']).tobytes()).hexdigest()==meta['rgb_sha256']
        videos.append(dict(number=number,source_path=s['source_path'],local_path=s['local_path'],duration_s=s['duration_s'],
            media_sha256=s['declared_sha256'],sheet=dest.relative_to(ROOT).as_posix(),sheet_sha256=sha(dest),
            sampled_frames=metadata,decoded_frames=decoded,seconds=time.monotonic()-began,
            official_question_ids=[q['id'] for q in s['questions']],content_reviewed=False))
        print(dict(source=number,samples=len(frames),seconds=time.monotonic()-began),flush=True)
        frames.clear()
    save(OUT/'actual_overviews14.json',dict(status='actual_RGB_verified_NOT_content_admission',videos=videos,
        sources=14,frames=sum(len(v['sampled_frames']) for v in videos),PNG_bytes=total,RSS_observed_bytes=peak,
        seconds=time.monotonic()-start,pretrained_loads=0,GPU_jobs=0))


if __name__=='__main__':main()
