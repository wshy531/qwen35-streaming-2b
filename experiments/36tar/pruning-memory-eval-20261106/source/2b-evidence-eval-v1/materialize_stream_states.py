"""Recover recent sampled frames by seeking, only with a no-skipped-second proof.

The frozen sampler admits at most one real sample per integer second. Its
committed sample count equalling floor(last PTS)+1 proves every prior second
was represented. Otherwise refuse this shortcut rather than substitute frames.
"""
import argparse,hashlib,json,math,os,sys,time
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
    run=R/'runs/evidence-stream-merged-20261102-01';assert read(run/'completed.json')['answers']==360
    out=R/'runs/evidence-stream-state-materials-20261102-01';assert not out.exists();out.mkdir();(out/'sampled_frames').mkdir()
    qs={q['id']:q for q in read(P/'panel_media_bound.json') if q['dataset']=='StreamingBench'}
    rows=[json.loads(x) for x in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    by={};verified=set();manifests=[];png_bytes=0;start=time.monotonic()
    for r in rows:
        key=r['id'];value=(r['visible_pairs'],r['source_state'])
        if key in by:assert by[key]==value
        by[key]=value
    assert len(by)==30
    for key,(pair_ids,state) in by.items():
        assert time.monotonic()-start<300
        q=qs[key];path=R/q['media_path'];name=q['source_id']
        if name not in verified:assert sha(path)==q['media_sha256'];verified.add(name)
        real_committed=2*state['complete_pairs']-state['synthetic_samples']
        assert real_committed==math.floor(state['last_committed_pts']+1e-9)+1,'full replay required: skipped-second proof unavailable'
        ordinals=[min(2*pi+j,real_committed-1) for pi in pair_ids for j in (0,1)]
        wanted=sorted(set(ordinals));images={};timestamps={}
        with av.open(str(path)) as c:
            stream=c.streams.video[0];stream.thread_count=1;origin=float(next(c.decode(stream)).time)
            c.seek(max(0,int((origin+min(wanted)-2)*av.time_base)),backward=True,any_frame=False)
            remaining=list(wanted)
            for frame in c.decode(stream):
                if frame.time is None:continue
                pts=float(frame.time)-origin
                if pts+1e-9<remaining[0]:continue
                target=remaining.pop(0)
                assert math.floor(pts+1e-9)==target,'seek missed first eligible integer-second frame'
                images[target]=canvas(frame.to_ndarray(format='rgb24'));timestamps[target]=pts
                if not remaining:break
            assert not remaining
        assert abs(timestamps[max(wanted)]-state['last_committed_pts'])<1e-7
        frames=[]
        for sample in wanted:
            pts=timestamps[sample];assert pts<=q['query_time_s']+1e-6
            dest=out/'sampled_frames'/f'{name}-{sample}.png';rgb=images[sample];digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            if dest.exists():
                with Image.open(dest) as im:assert hashlib.sha256(im.tobytes()).hexdigest()==digest
            else:Image.fromarray(rgb).save(dest);png_bytes+=dest.stat().st_size
            assert png_bytes<128*2**20
            frames.append(dict(sample_index=sample,frame_id=sample,frame_id_kind='sample_ordinal_not_original_decoded_frame_index',
                pts_s=pts,path=dest.relative_to(R).as_posix(),rgb_sha256=digest))
        manifests.append(dict(material_id=key,source_id=name,frames=frames,pair_sample_indices=ordinals,complete=True,
            query_time_s=q['query_time_s'],no_skipped_second_proof=True,
            derivation='same frozen canvas and integer-second sampler; source SHA and committed endpoint PTS; no original RGB hash claim'))
    (out/'materials.json').write_text(json.dumps(manifests,indent=2)+'\n')
    result=dict(materials=30,source_files=len(verified),unique_pngs=len(list((out/'sampled_frames').glob('*.png'))),png_bytes=png_bytes,
        seconds=time.monotonic()-start,GPU=0,API=0,downloads=0,no_skipped_second_proofs=30,original_RGB_SHA_comparison=False)
    (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
