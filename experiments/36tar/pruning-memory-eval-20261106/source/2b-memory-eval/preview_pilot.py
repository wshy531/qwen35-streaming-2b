"""Create query-prefix contact sheets for human review; not model sampling or scoring."""
import argparse
import json
import textwrap
import time
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--media',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sources=json.loads((args.media/'download_manifest.json').read_text())['files']
    source_names={x['source'] for x in sources}
    questions=[r for r in json.loads(args.manifest.read_text(encoding='utf-8'))
               if r['dataset']=='ovo' and r['source_id'] in source_names]
    deadline=time.monotonic()+120
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    except OSError:font=ImageFont.load_default()
    reports=[]
    for q in questions:
        if time.monotonic()>deadline:raise TimeoutError('Preview time budget exceeded')
        path=args.media/q['source_id']
        sheet=Image.new('RGB',(1152,580),(24,24,24))
        draw=ImageDraw.Draw(sheet)
        title=f"{q['id']} / {q['task']} / query={q['query_time_s']:.3f}s"
        draw.text((8,6),title,fill='white',font=font)
        for i,line in enumerate(textwrap.wrap(q['question'],width=110)):
            draw.text((8,28+18*i),line,fill='white',font=font)
        sampled=[]
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            stream.codec_context.thread_count=1
            first=next(container.decode(stream))
            origin=float(first.time)
            duration=float(container.duration/av.time_base) if container.duration else float(stream.duration*stream.time_base)
            if not 0<=q['query_time_s']<=duration+.05:
                raise ValueError('Query exceeds source duration')
            for index,fraction in enumerate((0,.2,.4,.6,.8,1)):
                target=origin+q['query_time_s']*fraction
                container.seek(max(0,int(target*av.time_base)),backward=True,any_frame=False)
                chosen=None
                for count,frame in enumerate(container.decode(stream)):
                    if count>=300 or time.monotonic()>deadline:raise TimeoutError('Seek decode budget exceeded')
                    if frame.time is None:continue
                    if float(frame.time)>target+1e-9:break
                    chosen=frame
                if chosen is None:raise ValueError('No arrived frame at preview target')
                timestamp=float(chosen.time)-origin
                if timestamp>q['query_time_s']+1e-9:raise ValueError('Future preview frame')
                image=ImageOps.contain(chosen.to_image(),(376,228))
                x=(index%3)*384;y=92+(index//3)*244
                sheet.paste(image,(x+(384-image.width)//2,y))
                draw.text((x+5,y+228),f'PTS {timestamp:.3f}s',fill='white',font=font)
                sampled.append(timestamp)
            source_info=dict(width=stream.width,height=stream.height,duration_s=duration,
                             fps=float(stream.average_rate) if stream.average_rate else None,first_pts_s=origin)
        filename=q['id'].replace(':','-')+'.jpg'
        sheet.save(args.out/filename,quality=88)
        reports.append(dict(id=q['id'],source=q['source_id'],query_time_s=q['query_time_s'],
            preview_pts_s=sampled,source_info=source_info,preview=filename,
            status='sparse_prefix_preview_only_not_answer_or_absence_verified'))
    (args.out/'preview_report.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(questions=len(reports),sources=len(source_names),future_preview_frames=0,
                         status='decode_and_preview_passed_human_evidence_review_pending'),indent=2))


if __name__=='__main__':main()
