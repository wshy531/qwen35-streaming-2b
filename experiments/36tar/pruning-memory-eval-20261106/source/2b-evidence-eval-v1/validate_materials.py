"""Bounded causal frame checks; catches missing, duplicate, future and changed images."""
import hashlib,json
from pathlib import Path
def validate(material,query_time,root,check_pixels=True):
    assert material['complete'] is True and material['frames']
    prior=-1.;seen=set();bytes_=0
    from PIL import Image
    for frame in material['frames']:
        t=frame['pts_s'];assert 0<=t<=query_time+1e-6 and t>prior
        assert frame['frame_id'] not in seen;seen.add(frame['frame_id']);prior=t
        path=(root/frame['path']).resolve();assert path.is_relative_to((root/'sampled_frames').resolve()) and path.is_file()
        bytes_+=path.stat().st_size
        if check_pixels:
            with Image.open(path) as im:
                assert im.size==(384,288) and im.mode=='RGB'
                assert hashlib.sha256(im.tobytes()).hexdigest()==frame['rgb_sha256']
    return dict(frames=len(seen),bytes=bytes_)
def main():
    import argparse,copy
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
    qs={q['id']:q for q in json.loads((P/'profile.json').read_text(encoding='utf-8'))}
    materials=json.loads((a.run/'materials.json').read_text());assert len(materials)==12
    results=[validate(m,qs[m['material_id']]['query_time_s'],a.run) for m in materials]
    faults=0
    for mutate in ('future','duplicate','escape','pixel'):
        m=copy.deepcopy(materials[0]);q=qs[m['material_id']]
        if mutate=='future':m['frames'][-1]['pts_s']=q['query_time_s']+2
        if mutate=='duplicate':m['frames'][-1]['frame_id']=m['frames'][0]['frame_id']
        if mutate=='escape':m['frames'][0]['path']='../outside.png'
        if mutate=='pixel':m['frames'][0]['rgb_sha256']='0'*64
        try:validate(m,q['query_time_s'],a.run)
        except AssertionError:faults+=1
    assert faults==4
    out=dict(materials=12,sampled_frames=sum(x['frames'] for x in results),png_bytes=sum(x['bytes'] for x in results),
             injected_faults_rejected=faults,GPU=0,API=0)
    (a.run/'material_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
