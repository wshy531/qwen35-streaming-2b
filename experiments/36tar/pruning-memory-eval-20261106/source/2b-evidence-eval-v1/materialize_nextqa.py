"""CPU-only bounded material export; reuse prior PNGs by source/frame hashes."""
import os,sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01';OUT=R/'runs/evidence-nextqa-materials-20261102-01'
sys.path.insert(0,str(R/'experiments/2b-learned-prune-main-v1'))
from protocol import runtime,read,sha,DATA
from audit_media import canvas
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and not OUT.exists()
    OUT.mkdir();(OUT/'sampled_frames').mkdir();start=time.monotonic()
    env=runtime();env['torch'].set_num_threads(1)
    audits={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
    panel=[q for q in read(P/'panel.json') if q['dataset']=='NExT-QA']
    existing={}
    for material in read(R/'runs/evidence-profile-220065/materials.json'):
        for f in material['frames']:
            existing[f['rgb_sha256']]=R/'runs/evidence-profile-220065'/f['path']
    materials=[];new_count=used_count=new_bytes=0
    for vid in sorted({q['source_id'] for q in panel}):
        assert time.monotonic()-start<240
        q=next(q for q in panel if q['source_id']==vid);path=R/q['media_path'];assert sha(path)==q['media_sha256']
        samples=audits[vid]['samples'];frames=[];wanted={f['frame_index']:f for f in samples};decoded={}
        missing=[f for f in samples if f['rgb_sha256'] not in existing]
        if missing:
            with env['av'].open(str(path)) as c:
                video=c.streams.video[0];video.thread_count=1
                for i,frame in enumerate(c.decode(video)):
                    if i in wanted and wanted[i]['rgb_sha256'] not in existing:decoded[i]=canvas(frame)
        for f in samples:
            digest=f['rgb_sha256']
            if digest in existing:
                dest=existing[digest];used_count+=1
                from PIL import Image
                with Image.open(dest) as image:assert hashlib.sha256(image.tobytes()).hexdigest()==digest
            else:
                image=decoded.pop(f['frame_index']);assert hashlib.sha256(image.tobytes()).hexdigest()==digest
                dest=OUT/'sampled_frames'/f'{vid}-{f["frame_index"]}.png';image.save(dest);existing[digest]=dest
                new_count+=1;new_bytes+=dest.stat().st_size;assert new_bytes<128*2**20
            assert f['pts_s']<=q['query_time_s']+1e-6
            frames.append(dict(frame_id=f['frame_index'],pts_s=f['pts_s'],path=dest.relative_to(R).as_posix(),rgb_sha256=digest))
        materials.append(dict(source_id=vid,frames=frames,complete=True,media_sha256=q['media_sha256']))
    (OUT/'materials.json').write_text(json.dumps(materials,indent=2)+'\n')
    result=dict(sources=len(materials),sampled_frames=sum(len(m['frames']) for m in materials),new_pngs=new_count,reused_pngs=used_count,new_bytes=new_bytes,
        seconds=time.monotonic()-start,GPU=0,API=0,downloads=0)
    (OUT/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
