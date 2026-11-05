"""Bounded real-PTS sampling and review sheets; no model or content verdicts."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import av
from PIL import Image,ImageDraw
import psutil

from inventory_local import ROOT,OUT
from schedule import Frame,Query,events

sys.path.insert(0,str(ROOT/'experiments/2b-memory-eval'))
from audit_sampling import canvas_bytes


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    rows=read(OUT/'existing6_EPM28_candidates_LOCAL_ONLY.json')
    groups=defaultdict(list)
    for row in rows:groups[row['source_path']].append(row)
    plan=dict(candidate_sha256=sha(OUT/'existing6_EPM28_candidates_LOCAL_ONLY.json'),
        sources=6,questions=28,fps=1,canvas_hw=[192,256],frames_per_page=24,
        overview_stride_sample_indices=5,max_buffered_RGB_frames=48,
        per_source_wall_seconds=240,total_wall_seconds=1500,RSS_cap_bytes=1024**3,
        PNG_cap_bytes=256*2**20,all_input_frames_hashed=True,
        review_scope='overview for localization, then actual1FPS pages; do not claim unviewed pages reviewed',
        model_outputs_used=False,model_loads=0,GPU_jobs=0,
        source_sha256=sha(Path(__file__)),canvas_sha256=sha(ROOT/'experiments/2b-memory-eval/audit_sampling.py'))
    save(OUT/'actual_sampling_plan.json',plan)
    folder=OUT/'sheets';folder.mkdir(exist_ok=False)
    started=time.monotonic();total_bytes=peak_rss=crop_checks=0;traces=[]
    for name,qs in groups.items():
        p=Path(qs[0]['local_path']);assert sha(p)==qs[0]['media_sha256']
        ident=Path(name).stem;begin=time.monotonic();frames=[];pages=[];overview=[]
        buffers={'full':[],'overview':[]};origin=previous=None;due=0.;decoded=0

        def flush(kind):
            nonlocal total_bytes,crop_checks
            buf=buffers[kind];height=math.ceil(len(buf)/4)*214
            sheet=Image.new('RGB',(1024,height),'white');draw=ImageDraw.Draw(sheet);locations=[]
            for k,(meta,rgb) in enumerate(buf):
                x,y=k%4*256,k//4*214
                draw.text((x+3,y+3),f"{meta['pts_s']:.3f}s id{meta['frame_id']}",fill='black')
                sheet.paste(Image.frombytes('RGB',(256,192),rgb),(x,y+22))
                locations.append(dict(**meta,crop_xyxy=[x,y+22,x+256,y+214]))
            target=overview if kind=='overview' else pages
            dest=folder/f'{ident}-{kind}-{len(target):02d}.png';sheet.save(dest);sheet.close()
            total_bytes+=dest.stat().st_size
            assert total_bytes<=plan['PNG_cap_bytes']
            with Image.open(dest) as reopened:
                for m in locations:
                    assert hashlib.sha256(reopened.crop(m['crop_xyxy']).tobytes()).hexdigest()==m['rgb_sha256']
                    crop_checks+=1
            target.append(dict(path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),frames=locations))
            buf.clear()

        with av.open(str(p)) as c:
            stream=c.streams.video[0];stream.codec_context.thread_count=1
            for i,frame in enumerate(c.decode(stream)):
                assert time.monotonic()-started<plan['total_wall_seconds']
                assert time.monotonic()-begin<plan['per_source_wall_seconds']
                assert frame.time is not None
                if origin is None:origin=float(frame.time)
                pts=float(frame.time)-origin;decoded+=1
                assert previous is None or pts>previous
                previous=pts
                if pts+1e-9<due:continue
                rgb=canvas_bytes(frame)
                m=dict(frame_id=i,pts_s=pts,rgb_sha256=hashlib.sha256(rgb).hexdigest())
                frames.append(m);buffers['full'].append((m,rgb))
                if (len(frames)-1)%5==0:buffers['overview'].append((m,rgb))
                due=math.floor(pts+1e-9)+1.
                peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
                assert peak_rss<plan['RSS_cap_bytes']
                for kind in buffers:
                    if len(buffers[kind])==24:flush(kind)
        for kind in buffers:
            if buffers[kind]:flush(kind)
        end_s=max(qs[0]['source_duration_s'],frames[-1]['pts_s'])
        schedule=list(events((Frame(f['frame_id'],f['pts_s']) for f in frames),
            [Query(q['id'],q['query_time_s']) for q in sorted(qs,key=lambda q:(q['query_time_s'],q['id']))],end_s=end_s))
        checks=[e for e in schedule if e['kind']=='query']
        for q in checks:
            assert q['committed_through_s'] is not None and q['committed_through_s']<=q['query_time_s']
            committed=frames[:q['committed_pairs']*2]
            q['committed_frames_sha256']=hashlib.sha256(json.dumps(committed,sort_keys=True).encode()).hexdigest()
            q['last_committed_frame_id']=committed[-1]['frame_id']
        trace=dict(source_path=name,media_path=str(p.relative_to(ROOT)),media_sha256=sha(p),
            pts_origin_s=origin,duration_s=end_s,frames=frames,pages=pages,overview_pages=overview,
            query_snapshots=checks,decoded_frames=decoded,seconds=time.monotonic()-begin)
        save(OUT/f'sampling-{ident}.json',trace);traces.append(trace)
        print(dict(source=name,frames=len(frames),full_pages=len(pages),overview_pages=len(overview),seconds=trace['seconds']),flush=True)
    save(OUT/'actual_sampling_all6.json',dict(videos=traces))
    save(OUT/'actual_sampling_receipt.json',dict(status='actual_PTS_RGB_and_query_cutoffs_verified_NOT_content_review',
        sources=6,questions=28,frames=sum(len(t['frames']) for t in traces),
        pages=sum(len(t['pages'])+len(t['overview_pages']) for t in traces),
        crop_SHA_checks=crop_checks,PNG_bytes=total_bytes,peak_process_RSS_bytes=peak_rss,
        seconds=time.monotonic()-started,plan_sha256=sha(OUT/'actual_sampling_plan.json'),
        GPU_jobs=0,model_loads=0,content_verdicts=0))
    print(read(OUT/'actual_sampling_receipt.json'),flush=True)


if __name__=='__main__':main()
