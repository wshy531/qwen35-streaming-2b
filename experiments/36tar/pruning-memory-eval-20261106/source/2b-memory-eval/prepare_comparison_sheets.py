"""Precommitted comparison annotation sheets, no holdout images/model outputs."""
import hashlib
import json
from pathlib import Path
import time

import av
from PIL import Image,ImageDraw,ImageFont,ImageOps

from audit_sampling import canvas_bytes
from prepare_grounded_review import select_witnesses


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    out=run/'gqa_comparison_sheets';out.mkdir(exist_ok=False)
    rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    traces=json.loads((run/'gqa48_actual_index.json').read_text(encoding='utf-8'))['videos']
    plan=[]
    for trace in traces:
        if trace['split']!='comparison':continue
        qs=[r for r in rows if r['dataset']=='nextgqa' and r['source_path']==trace['source_path']]
        intervals=sorted({tuple(v) for q in qs for v in q['evidence_intervals']})
        chosen=select_witnesses(trace['frames_1fps'],intervals)
        plan.append(dict(source_path=trace['source_path'],local_path=trace['local_path'],
            source_sha256=trace['source_sha256'],query_time_s=trace['max_query_s'],
            author_intervals=intervals,questions=qs,frames=chosen))
    assert len(plan)==24
    (out/'selection_plan.json').write_text(json.dumps(dict(
        purpose='author_guided_annotation_only_not_retrieval_or_model_input',
        selection_committed_before_viewing=True,holdout_sources=0,sources=plan),indent=2)+'\n',encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);receipts=[]
    for item in plan:
        start=time.monotonic();sheet=Image.new('RGB',(1600,1080),(18,18,18));draw=ImageDraw.Draw(sheet)
        draw.text((8,3),item['source_path']+' | comparison annotation only | actual 1FPS',font=font,fill='white')
        wanted={f['frame_id']:(i,f) for i,f in enumerate(item['frames'])};seen=[];origin=None
        with av.open(item['local_path']) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            for frame_id,frame in enumerate(container.decode(stream)):
                if time.monotonic()-start>120:raise TimeoutError('preview source budget')
                if origin is None:origin=float(frame.time)
                if frame_id not in wanted:continue
                i,expected=wanted[frame_id];pts=float(frame.time)-origin
                if pts>item['query_time_s'] or abs(pts-expected['pts_s'])>1e-7 or hashlib.sha256(canvas_bytes(frame)).hexdigest()!=expected['rgb_sha256']:
                    raise ValueError('preview differs from actual source prefix')
                x=(i%4)*400;y=30+(i//4)*260
                image=ImageOps.contain(frame.to_image(),(396,232));sheet.paste(image,(x+(400-image.width)//2,y))
                draw.text((x+4,y+234),f'id {frame_id} PTS {pts:.3f}',font=font,fill='white')
                seen.append(dict(**expected,cell=i,hash_verified=True))
                if len(seen)==len(wanted):break
        if len(seen)!=len(wanted):raise ValueError('missing preview frame')
        name='nextgqa-'+Path(item['source_path']).name+'.jpg';sheet.save(out/name,quality=90)
        receipts.append(dict(source_path=item['source_path'],image=name,frames=seen,elapsed_s=time.monotonic()-start))
        print(json.dumps(dict(source=item['source_path'],frames=len(seen),hashes_verified=True)),flush=True)
        (out/'review_receipts.json').write_text(json.dumps(dict(status='generated_not_yet_visually_reviewed',sources=receipts),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
