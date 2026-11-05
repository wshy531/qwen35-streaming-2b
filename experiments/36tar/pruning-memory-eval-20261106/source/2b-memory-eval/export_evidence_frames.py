"""Export a few original-resolution arrived frames for evidence review only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import av


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--times',type=float,nargs='+',required=True)
    p.add_argument('--query-time',type=float,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if len(a.times)>16 or any(not 0<=t<=a.query_time for t in a.times):
        raise ValueError('At most 16 arrived review targets required')
    a.out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+120
    records=[]
    with av.open(str(a.video)) as c:
        s=c.streams.video[0]
        s.codec_context.thread_count=1
        first=next(c.decode(s))
        origin=float(first.time)
        del first
        for i,t in enumerate(a.times):
            c.seek(max(0,int((origin+t)*av.time_base)),backward=True,any_frame=False)
            chosen=None
            for n,f in enumerate(c.decode(s)):
                if n>400 or time.monotonic()>deadline:raise TimeoutError('Review decode budget')
                if f.time is None:continue
                if float(f.time)>origin+t+1e-9:break
                chosen=f
            if chosen is None:raise ValueError('No arrived frame')
            image=chosen.to_image()
            name=f'{i:02d}-{float(chosen.time)-origin:.3f}s.jpg'
            image.save(a.out/name,quality=95)
            records.append(dict(target_s=t,actual_pts_s=float(chosen.time)-origin,image=name,
                width=image.width,height=image.height,rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
            del chosen,image
    (a.out/'frames.json').write_text(json.dumps(dict(source=str(a.video),frames=records,
        purpose='single_reviewer_original_resolution_annotation_not_model_input'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(frames=len(records),out=str(a.out))))


if __name__=='__main__':main()
