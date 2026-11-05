"""Preselect bounded annotation-only sheets from exact previously sampled frames.

Ground-truth intervals guide human review only. These selections MUST NOT be fed
to memory policies or described as automatic retrieval performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps
from audit_sampling import canvas_bytes


def select_witnesses(frames, intervals, limit=16):
    chosen={}
    priorities=[]
    for start,end in intervals:
        inside=[f for f in frames if start<=f['pts_s']<=end]
        if not inside:
            continue
        priorities.append(inside[len(inside)//2])
    priorities += [frames[0],frames[len(frames)//2],frames[-1]]
    for start,end in intervals:
        inside=[f for f in frames if start<=f['pts_s']<=end]
        if inside:
            priorities += [inside[0],inside[-1]]
    # Add near-boundary context while preserving a deterministic cap.
    for start,end in intervals:
        for target in (max(0,start-1),end+1):
            priorities.append(min(frames,key=lambda f:abs(f['pts_s']-target)))
    for frame in priorities:
        if len(chosen)>=limit:
            break
        chosen[frame['frame_id']]=frame
    return sorted(chosen.values(),key=lambda f:f['pts_s'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--followup-spec',type=Path,
                        help='Predeclared source/frame IDs and concrete annotation gap; no new sampling')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    evidence=root/'runs/development_audit_20261017_02'
    all_rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    rows=[r for r in all_rows if r['dataset']=='nextgqa' and r['split']=='development']
    traces={v['source_path']:v for v in json.loads((evidence/'sampling_gqa12_1fps.json').read_text(encoding='utf-8'))['videos']}
    media={r['source_path']:r for r in json.loads((evidence/'gqa_media_inventory.json').read_text(encoding='utf-8'))['files']}
    plan=[]
    for source in dict.fromkeys(r['source_path'] for r in rows):
        questions=[r for r in rows if r['source_path']==source]
        intervals=sorted({tuple(span) for r in questions for span in r['evidence_intervals']})
        frames=select_witnesses(traces[source]['frames'],intervals)
        query=questions[0]['query_time_s']
        assert all(f['pts_s']<=query+1e-9 for f in frames)
        plan.append(dict(source_path=source,source_sha256=media[source]['sha256'],
            local_path=media[source]['local_path'],query_time_s=query,
            question_ids=[r['id'] for r in questions],author_intervals=intervals,frames=frames))
    if args.followup_spec:
        spec=json.loads(args.followup_spec.read_text(encoding='utf-8'))
        if not 1<=len(spec)<=2 or len({x['source_path'] for x in spec})!=len(spec):
            raise ValueError('at most two distinct follow-up sources')
        selected=[]
        for request in spec:
            if not request.get('reason') or not 1<=len(request['frame_ids'])<=16:
                raise ValueError('bounded follow-up requires a concrete gap')
            item=next(x for x in plan if x['source_path']==request['source_path'])
            ids=set(request['frame_ids'])
            frames=[f for f in traces[item['source_path']]['frames'] if f['frame_id'] in ids]
            if len(frames)!=len(request['frame_ids']):
                raise ValueError('follow-up must use unique actual sampled frames')
            selected.append(dict(item,frames=frames,followup_reason=request['reason']))
        plan=selected
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'selection_plan.json').write_text(json.dumps(dict(
        purpose='annotation_only_author_guided_not_model_input_or_retrieval_test',
        selected_before_visual_review=True,sources=plan),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    receipts=[]
    for item in plan:
        start=time.monotonic()
        sheet=Image.new('RGB',(1600,1080),(18,18,18));draw=ImageDraw.Draw(sheet)
        draw.text((8,3),f'{item["source_path"]} | exact 1FPS frames | annotation only',font=font,fill='white')
        seen=[]
        with av.open(item['local_path']) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            wanted={f['frame_id']:(i,f) for i,f in enumerate(item['frames'])}
            origin=None
            for frame_id,chosen in enumerate(container.decode(stream)):
                if time.monotonic()-start>120:
                    raise TimeoutError('bounded per-source review budget')
                if chosen.time is None:
                    raise ValueError('missing review PTS')
                if origin is None:
                    origin=float(chosen.time)
                if frame_id not in wanted:
                    continue
                i,frame_info=wanted[frame_id]
                if abs(float(chosen.time)-origin-frame_info['pts_s'])>1e-7 or \
                   hashlib.sha256(canvas_bytes(chosen)).hexdigest()!=frame_info['rgb_sha256']:
                    raise ValueError('review frame does not match actual sampled RGB')
                x=(i%4)*400;y=30+(i//4)*260
                image=ImageOps.contain(chosen.to_image(),(396,232))
                sheet.paste(image,(x+(400-image.width)//2,y))
                draw.text((x+4,y+234),f'id {frame_info["frame_id"]} PTS {frame_info["pts_s"]:.3f}',font=font,fill='white')
                seen.append(dict(**frame_info,cell=i,hash_verified=True))
                del chosen,image
                if len(seen)==len(wanted):
                    break
            if len(seen)!=len(wanted):
                raise ValueError('selected review frames missing at EOF')
        name='nextgqa-'+Path(item['source_path']).name+'.jpg'
        sheet.save(args.out/name,quality=90)
        receipts.append(dict(source_path=item['source_path'],image=name,
            frames=seen,elapsed_s=time.monotonic()-start))
        print(json.dumps(dict(source=item['source_path'],frames=len(seen),hashes_verified=True)),flush=True)
    (args.out/'review_receipts.json').write_text(json.dumps(dict(
        status='sheets_generated_not_yet_visually_reviewed',sources=receipts),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
