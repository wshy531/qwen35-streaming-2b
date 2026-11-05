"""Uniform single candidate-cohort check at exact model canvas pixels."""
import hashlib
import time

import av
from PIL import Image,ImageDraw,ImageFont

from audit_sampling import canvas_bytes
from prepare_natural_v2 import ROOT,OUT,read,save,sha


def main():
    start=time.monotonic();plan=read(OUT/'model_canvas_review_plan.json')
    assert sha(OUT/'candidate77.json')==plan['candidate77_sha256']
    folder=OUT/'model_canvas_sheets';folder.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);result=[]
    for item in plan['sources']:
        if time.monotonic()-start>570:raise TimeoutError('canvas review preparation budget')
        path=ROOT/item['media_path'];assert sha(path)==item['media_sha256']
        wanted={f['frame_id']:(i,f) for i,f in enumerate(item['frames'])}
        sheets=[Image.new('RGB',(1024,890),(15,15,15)) for _ in range((len(wanted)+15)//16)]
        for n,sheet in enumerate(sheets):ImageDraw.Draw(sheet).text((4,4),item['source_path']+f' | exact192x256 RGB | page{n+1}',font=font,fill='white')
        seen=[];origin=None
        with av.open(str(path)) as c:
            stream=c.streams.video[0];stream.codec_context.thread_count=1
            for fid,frame in enumerate(c.decode(stream)):
                if origin is None:origin=float(frame.time)
                if fid not in wanted:continue
                i,expected=wanted[fid];rgb=canvas_bytes(frame);pts=float(frame.time)-origin
                assert hashlib.sha256(rgb).hexdigest()==expected['rgb_sha256'] and abs(pts-expected['pts_s'])<1e-7
                assert pts<=item['query_time_s']
                page=i//16;cell=i%16;x=(cell%4)*256;y=30+(cell//4)*214
                sheets[page].paste(Image.frombytes('RGB',(256,192),rgb),(x,y))
                ImageDraw.Draw(sheets[page]).text((x+2,y+194),f'id {fid} PTS {pts:.3f}',font=font,fill='white')
                seen.append(dict(frame_id=fid,pts_s=pts,rgb_sha256=expected['rgb_sha256'],page=page+1,cell=cell))
                if len(seen)==len(wanted):break
        assert len(seen)==len(wanted)
        files=[]
        for n,sheet in enumerate(sheets):
            p=folder/f'nextgqa-{item["source_path"].split("/")[-1]}-p{n+1}.png';sheet.save(p)
            files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
        result.append(dict(source_path=item['source_path'],images=files,frames=seen))
        save(folder/'receipts.json',dict(status='generated_not_yet_visually_reviewed',sources=result))
        assert sum(p.stat().st_size for p in folder.iterdir() if p.is_file())<plan['max_disk_bytes']
    save(folder/'completed.json',dict(status='complete',sources=len(result),pages=sum(len(x['images']) for x in result),
        frames=sum(len(x['frames']) for x in result),seconds=time.monotonic()-start,model_loads=0,
        rendering='native256x192 RGB pasted without rescaling; lossless PNG',candidate77_sha256=plan['candidate77_sha256']))
    print(read(folder/'completed.json'))


if __name__=='__main__':main()
