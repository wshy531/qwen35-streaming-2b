"""One initial bounded annotation pass for new78 only, never model inputs."""
import hashlib
import json
import time

import av
from PIL import Image,ImageDraw,ImageFont,ImageOps

from audit_sampling import canvas_bytes
from prepare_grounded_review import select_witnesses
from prepare_natural_v2 import ROOT,OUT,read,save,sha


def main():
    start=time.monotonic();rows=read(OUT/'additional78.json')
    traces={t['source_path']:t for t in read(OUT/'sampling_1fps.json')}
    folder=OUT/'additional78_sheets';folder.mkdir(exist_ok=False)
    grouped={}
    for r in rows:grouped.setdefault(r['source_path'],[]).append(r)
    plan=[]
    for source,qs in grouped.items():
        trace=traces[source];intervals=sorted({tuple(x) for q in qs for x in q['evidence_intervals']})
        frames=select_witnesses(trace['frames'],intervals,limit=32)
        plan.append(dict(source_path=source,media_path=qs[0]['media_path'],media_sha256=qs[0]['media_sha256'],
            query_time_s=qs[0]['query_time_s'],question_ids=[q['id'] for q in qs],frames=frames))
    save(folder/'initial_review_plan.json',dict(status='locked_before_new_sheet_view',sources=plan,
        maximum_frames_per_source=32,maximum_pages_per_source=2,annotation_only=True,
        old71_dispositions_may_be_changed=False,old_failed_questions_may_be_reopened=False,
        maximum_cpu_seconds=600,maximum_disk_bytes=32*2**20,model_loads=0,holdout_sources=0,
        reason='Old sheets were selected for old71; this is the first image-review batch for additional78 questions, not another round on old ambiguities.'))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);receipts=[]
    for item in plan:
        if time.monotonic()-start>570:raise TimeoutError('annotation batch time limit')
        path=ROOT/item['media_path'];assert sha(path)==item['media_sha256']
        wanted={f['frame_id']:(i,f) for i,f in enumerate(item['frames'])}
        sheets=[Image.new('RGB',(1600,1080),(18,18,18)) for _ in range((len(wanted)+15)//16)]
        for page,sheet in enumerate(sheets):
            ImageDraw.Draw(sheet).text((8,3),item['source_path']+f' | new78 initial annotation | 1FPS | page{page+1}',font=font,fill='white')
        seen=[];origin=None
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            for frame_id,frame in enumerate(container.decode(stream)):
                if time.monotonic()-start>590:raise TimeoutError('annotation decoding budget')
                if origin is None:origin=float(frame.time)
                if frame_id not in wanted:continue
                i,expected=wanted[frame_id];pts=float(frame.time)-origin
                assert abs(pts-expected['pts_s'])<1e-7 and pts<=item['query_time_s']
                assert hashlib.sha256(canvas_bytes(frame)).hexdigest()==expected['rgb_sha256']
                page=i//16;cell=i%16;x=(cell%4)*400;y=30+(cell//4)*260
                im=ImageOps.contain(frame.to_image(),(396,232));sheets[page].paste(im,(x+(400-im.width)//2,y))
                ImageDraw.Draw(sheets[page]).text((x+4,y+234),f'id {frame_id} PTS {pts:.3f}',font=font,fill='white')
                seen.append(dict(**expected,page=page+1,cell=cell,hash_verified=True,source_path=item['source_path']))
                if len(seen)==len(wanted):break
        assert len(seen)==len(wanted)
        images=[]
        for i,sheet in enumerate(sheets):
            target=folder/f'nextgqa-{item["source_path"].split("/")[-1]}-p{i+1}.jpg'
            sheet.save(target,quality=90);images.append(dict(path=str(target.relative_to(ROOT)),sha256=sha(target)))
        receipts.append(dict(source_path=item['source_path'],images=images,frames=seen))
        save(folder/'receipts.json',dict(status='generated_not_yet_visually_reviewed',sources=receipts))
        assert sum(p.stat().st_size for p in folder.iterdir() if p.is_file())<32*2**20
        print(json.dumps(dict(source=item['source_path'],frames=len(seen),pages=len(images))),flush=True)
    save(folder/'completed.json',dict(status='complete_initial_new78_annotation_images',sources=len(receipts),
        frames=sum(len(s['frames']) for s in receipts),pages=sum(len(s['images']) for s in receipts),
        seconds=time.monotonic()-start,additional78_sha256=sha(OUT/'additional78.json'),
        existing71_re_adjudicated=False,model_loads=0,holdout_images=0))


if __name__=='__main__':main()
