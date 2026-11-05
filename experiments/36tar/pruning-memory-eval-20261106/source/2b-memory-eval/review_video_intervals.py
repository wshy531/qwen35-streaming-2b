"""Bounded CPU contact sheets for evidence annotation, never a model sampler."""
import argparse
import json
import math
import time
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--step', type=float, required=True)
    p.add_argument('--query-time', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not 0 <= a.start <= a.end <= a.query_time or a.step <= 0:
        p.error('Require 0 <= start <= end <= query-time and positive step')
    targets = [a.start + i*a.step for i in range(math.floor((a.end-a.start)/a.step)+1)]
    if len(targets) > 160:
        p.error('At most 160 review targets per invocation')
    a.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic()+120
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    report = []
    with av.open(str(a.video)) as c:
        stream = c.streams.video[0]
        stream.codec_context.thread_count = 1
        first = next(c.decode(stream))
        origin = float(first.time)
        del first
        for i, target in enumerate(targets):
            if i % 16 == 0:
                sheet = Image.new('RGB', (1600, 1080), (18, 18, 18))
                draw = ImageDraw.Draw(sheet)
                draw.text((8, 4), f'{a.video.name} / review only / query <= {a.query_time:.3f}s', font=font, fill='white')
            absolute = origin+target
            c.seek(max(0, int(absolute*av.time_base)), backward=True, any_frame=False)
            chosen = None
            for count, frame in enumerate(c.decode(stream)):
                if count >= 400 or time.monotonic() >= deadline:
                    raise TimeoutError('Review decode budget exhausted')
                if frame.time is None:
                    continue
                if float(frame.time) > absolute+1e-9:
                    break
                chosen = frame
            if chosen is None:
                raise ValueError(f'No arrived frame at {target}')
            pts = float(chosen.time)-origin
            if pts > a.query_time+1e-9:
                raise ValueError('Future evidence frame')
            x = (i % 4)*400
            y = 30+((i % 16)//4)*260
            im = ImageOps.contain(chosen.to_image(), (396, 232))
            sheet.paste(im, (x+(400-im.width)//2, y))
            draw.text((x+5, y+233), f'PTS {pts:.3f}s', font=font, fill='white')
            page = f'page-{i//16:02d}.jpg'
            report.append(dict(target_s=target, pts_s=pts, page=page, cell=i % 16))
            del chosen, im
            if i % 16 == 15 or i == len(targets)-1:
                sheet.save(a.out/page, quality=90)
                del sheet, draw
    (a.out/'frames.json').write_text(json.dumps(dict(source=str(a.video), query_time_s=a.query_time,
        frames=report, use='review_only_not_model_input'), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(frames=len(report), pages=math.ceil(len(report)/16), future_frames=0, out=str(a.out))))


if __name__ == '__main__':
    main()
