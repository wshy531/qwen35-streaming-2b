"""Prepare every already-frozen sampled frame for the pending early25 audit."""
import hashlib,json,time
from pathlib import Path
import av
from PIL import Image,ImageDraw,ImageFont
from audit_sampling import canvas_bytes
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/2b_natural_eval_20261020_01'
OUT=ROOT/'runs/early25_full_prefix_review_20261020_01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    pending=ROOT/'runs/memory_task_alignment_20261019_01/natural_early_FT25_pending.json';qs=read(pending)
    requests={r['id']:r for r in read(RUN/'requests.json')};media={r['remote_path']:r for r in read(RUN/'media_manifest.json')}
    grouped={}
    for q in qs:grouped.setdefault(q['source_path'],[]).append(q)
    OUT.mkdir(exist_ok=False);(OUT/'sheets').mkdir()
    save(OUT/'plan.json',dict(status='fixed_before_new_full_prefix_visual_review',questions=25,sources=20,original_pending_sha256=sha(pending),requests_sha256=sha(RUN/'requests.json'),input_identity='existing1FPS actual RGB/PTS, all sampled prefix frames, no annotation-driven frame selection',pagesize=24,max_disk_bytes=100*2**20,seconds=300,one_full_prefix_review=True,old_necessity_proven=False,old_08B_outputs_already_open_but_not_used_for_curation=True,new_2B_answers_not_opened_for_curation=True))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);start=time.monotonic();traces=[]
    for source,rows in grouped.items():
        if time.monotonic()-start>300:raise TimeoutError('bounded CPU image preparation')
        request=requests[rows[0]['id']];specs=request['model_visible']['frames'];m=media[request['media_path']];p=ROOT/m['local_path']
        if sha(p)!=request['media_sha256']:raise ValueError('media SHA drift')
        if any(requests[q['id']]['model_visible']['frames']!=specs for q in rows):raise ValueError('prefix mismatch')
        pages=[Image.new('RGB',(1024,1300),(16,16,16)) for _ in range((len(specs)+23)//24)]
        for j,page in enumerate(pages):ImageDraw.Draw(page).text((3,3),source+f' | FULL fixed1FPS prefix | p{j+1}',font=font,fill='white')
        wanted={f['frame_id']:(i,f) for i,f in enumerate(specs)};seen=[];origin=None
        with av.open(str(p)) as c:
            st=c.streams.video[0];st.codec_context.thread_count=1
            for fid,frame in enumerate(c.decode(st)):
                if origin is None:origin=float(frame.time)
                if fid not in wanted:continue
                i,f=wanted[fid];rgb=canvas_bytes(frame);pts=float(frame.time)-origin
                if hashlib.sha256(rgb).hexdigest()!=f['rgb_sha256'] or abs(pts-f['pts_s'])>1e-7:raise ValueError('actual frame drift')
                page=pages[i//24];cell=i%24;x=cell%4*256;y=26+cell//4*210
                page.paste(Image.frombytes('RGB',(256,192),rgb),(x,y));ImageDraw.Draw(page).text((x+2,y+193),f'{pts:.3f}s / f{fid}',font=font,fill='white')
                seen.append(dict(f,page=i//24+1,cell=cell))
                if len(seen)==len(specs):break
        if len(seen)!=len(specs):raise ValueError('incomplete prefix images')
        images=[]
        for j,page in enumerate(pages):
            dst=OUT/'sheets'/f'{source.split("/")[-1]}-p{j+1:02d}.png';page.save(dst);images.append(dict(path=dst.relative_to(ROOT).as_posix(),sha256=sha(dst)))
        traces.append(dict(source_path=source,questions=rows,media=m,frames=seen,pages=images,all_pages_reviewed=False))
        save(OUT/'image_receipts.json',traces)
        if sum(f.stat().st_size for f in (OUT/'sheets').iterdir())>100*2**20:raise MemoryError('100MiB image disk budget')
    save(OUT/'render_receipt.json',dict(status='generated_not_yet_reviewed',sources=len(traces),frames=sum(len(t['frames']) for t in traces),pages=sum(len(t['pages']) for t in traces),seconds=time.monotonic()-start,GPU_used=False))
    print(read(OUT/'render_receipt.json'))
if __name__=='__main__':main()
