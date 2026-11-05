"""Seek to exact recorded PTS after generation; never substitute nearby frames."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
sys.path.insert(0,str(R/'experiments/streamingbench-interface-v1'))
from source import canvas
import av
from PIL import Image
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    done=read(a.run/'completed.json');assert done['answers']==360
    out=R/'runs/evidence-stream-materials-20261102-01';assert not out.exists();out.mkdir();(out/'sampled_frames').mkdir()
    qs={q['id']:q for q in read(P/'panel_media_bound.json') if q['dataset']=='StreamingBench'}
    rows=[json.loads(x) for x in (a.run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    used={}
    for row in rows:
        selected=row['visible_pairs']
        if row['id'] in used:assert used[row['id']]==selected
        used[row['id']]=selected
    manifests=[];start=time.monotonic();pngs=bytes_=0
    for source in done['checked_sources']:
        assert time.monotonic()-start<300
        name=source['source_id'];q=next(q for q in qs.values() if q['source_id']==name);path=R/q['media_path']
        assert sha(path)==source['media_sha256']==q['media_sha256']
        provenance={p['pair']:p for p in source['pair_provenance']};wanted={};perquery={}
        for task in [q for q in qs.values() if q['source_id']==name]:
            key=task['id'];perquery[key]=[]
            for pi in used[key]:
                pair=provenance[pi]
                for side in ('first','second'):
                    frame_id=pair[side+'_frame'];pts=pair[side+'_pts'];assert pts<=task['query_time_s']+1e-6
                    wanted[frame_id]=pts;perquery[key].append(frame_id)
        remaining=dict(wanted);files={}
        with av.open(str(path)) as c:
            stream=c.streams.video[0];stream.thread_count=1
            first=next(c.decode(stream));origin=float(first.time)
            absolute_min=origin+min(wanted.values())
            c.seek(max(0,int((absolute_min-2)*av.time_base)),backward=True,any_frame=False)
            for frame in c.decode(stream):
                if frame.time is None:continue
                pts=float(frame.time)-origin
                match=next((i for i,t in remaining.items() if abs(t-pts)<1e-7),None)
                if match is not None:
                    rgb=canvas(frame.to_ndarray(format='rgb24'));dest=out/'sampled_frames'/f'{name}-{match}.png'
                    Image.fromarray(rgb).save(dest);pngs+=1;bytes_+=dest.stat().st_size;assert bytes_<128*2**20
                    files[match]=dict(frame_id=match,pts_s=pts,path=dest.relative_to(R).as_posix(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                    del remaining[match]
                if not remaining:break
                if pts>max(wanted.values())+1:break
        assert not remaining,(name,remaining) # no nearest-frame or dropped-frame fallback
        for key,ids in perquery.items():
            unique=list(dict.fromkeys(ids));manifests.append(dict(material_id=key,source_id=name,frames=[files[i] for i in unique],complete=True,
                query_time_s=qs[key]['query_time_s'],derivation='exact recorded pair PTS, same frozen canvas, source SHA; raw RGB was not hashed in the original worker'))
    assert len(manifests)==30
    (out/'materials.json').write_text(json.dumps(manifests,indent=2)+'\n')
    result=dict(materials=30,exported_unique_frames=pngs,png_bytes=bytes_,seconds=time.monotonic()-start,GPU=0,API=0,downloads=0,
                exact_recorded_PTS=True,no_original_pixel_hash_claim=True)
    (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
