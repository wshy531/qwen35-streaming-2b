"""Single-thread 1FPS dense evidence of three preselected local sources, no model."""
import hashlib,json,math,sys,time
from pathlib import Path
import av,psutil
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from audit_sampling import canvas_bytes
OUT=ROOT/'runs/2b_long_data_v2_prepare_20261021_01'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def main():
    receipt=json.loads((ROOT/'runs/2b_spatial_time_prepare_20261021_01/actual_overviews14.json').read_text(encoding='utf-8'))
    sources=receipt['videos'][:3];assert len(sources)==3 and all(s['source_path'].startswith('Ego4D/') for s in sources)
    OUT.mkdir(exist_ok=False);(OUT/'sheets').mkdir()
    save(OUT/'sampling_plan.json',dict(source_paths=[s['source_path'] for s in sources],selection='first3 predeclared overview priority; no output-based selection',
        fps=1,canvas=[192,256],max_buffered_frames=24,RSS_cap_bytes=2**30,PNG_cap_bytes=128*2**20,
        total_seconds=900,pretrained_loads=0,GPU_jobs=0,new_question_labels=0))
    start=time.monotonic();total=peak=0;results=[]
    for number,s in enumerate(sources):
        p=Path(s['local_path']);assert sha(p)==s['media_sha256']
        pending=[];metadata=[];pages=[];origin=previous=None;due=0.;decoded=0
        def flush():
            nonlocal total
            if not pending:return
            image=Image.new('RGB',(1024,math.ceil(len(pending)/4)*214),'white');d=ImageDraw.Draw(image);entries=[]
            for j,(m,rgb) in enumerate(pending):
                x,y=j%4*256,j//4*214;d.text((x+2,y+3),f'{m["pts_s"]:.3f}s id{m["frame_id"]}',fill='black')
                image.paste(Image.frombytes('RGB',(256,192),rgb),(x,y+22));entries.append(dict(**m,crop=[x,y+22,x+256,y+214]))
            dest=OUT/'sheets'/f'{number:02d}-{len(pages):02d}.png';image.save(dest);image.close()
            total+=dest.stat().st_size;assert total<128*2**20
            with Image.open(dest) as img:
                for m in entries:assert hashlib.sha256(img.crop(m['crop']).tobytes()).hexdigest()==m['rgb_sha256']
            pages.append(dict(path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),frames=entries,actually_viewed=False))
            pending.clear()
        with av.open(str(p)) as c:
            stream=c.streams.video[0];stream.codec_context.thread_count=1
            for i,frame in enumerate(c.decode(stream)):
                assert time.monotonic()-start<900 and frame.time is not None
                if origin is None:origin=float(frame.time)
                pts=float(frame.time)-origin;assert previous is None or pts>previous;previous=pts;decoded+=1
                if pts+1e-9<due:continue
                rgb=canvas_bytes(frame);m=dict(frame_id=i,pts_s=pts,rgb_sha256=hashlib.sha256(rgb).hexdigest());metadata.append(m)
                pending.append((m,rgb));assert len(pending)<=24
                peak=max(peak,psutil.Process().memory_info().rss);assert peak<2**30
                if len(pending)==24:flush()
                due=math.floor(pts+1e-9)+1.
        flush();assert sha(p)==s['media_sha256']
        results.append(dict(source_path=s['source_path'],local_path=str(p),media_sha256=s['media_sha256'],duration_s=s['duration_s'],
            frames=metadata,pages=pages,decoded_frames=decoded,admitted_questions=0))
        print(dict(source=number,frames=len(metadata),pages=len(pages)),flush=True)
    save(OUT/'actual_dense3.json',dict(status='actual_RGB_verified_NOT_dense_content_reviewed',sources=results,
        frames=sum(len(s['frames']) for s in results),pages=sum(len(s['pages']) for s in results),PNG_bytes=total,
        RSS_observed_bytes=peak,seconds=time.monotonic()-start,GPU_jobs=0,new_question_labels=0))


if __name__=='__main__':main()
