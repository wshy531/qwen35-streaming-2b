"""Read three authorized profile media, freeze first eight causal sampled frames."""
import hashlib,json
from pathlib import Path
import numpy as np
from video_source import CausalPairs
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*2**20):h.update(block)
    return h.hexdigest()
def main():
    tasks=json.loads((PREP/'profile.json').read_text(encoding='utf-8'));seen=set();rows=[]
    traces=json.loads((PREP/'traces.json').read_text(encoding='utf-8'))
    for q in tasks:
        key=q['suite']+'|'+q['source_id']
        if key in seen:continue
        seen.add(key);spec=traces[key];path=Path(spec['media_path'])
        if not path.is_absolute():path=ROOT/path
        assert sha(path)==q['media_sha256']
        frames=[];samples=[]
        def commit(a,b,padding=False):
            assert not padding
            for packet in (a,b):
                assert packet.rgb.shape==(280,392,3)
                frames.append(packet.rgb)
                samples.append(dict(frame_id=packet.frame_id,pts_s=packet.timestamp))
        driver=CausalPairs(path,commit)
        try:state=driver.advance(7.2)
        finally:driver.close()
        assert len(frames)==8
        if q['suite']=='D18':
            for sample,reference in zip(samples,spec['sampled_trace'][:8]):
                assert sample['frame_id']==reference['frame_id'] and abs(sample['pts_s']-reference['pts_s'])<1e-6
        dest=PREP/f'cpu_frames_{len(rows)}.npz'
        np.savez_compressed(dest,frames=np.stack(frames))
        rows.append(dict(key=key,media_sha256=q['media_sha256'],samples=samples,
            frames_file=dest.name,frames_sha256=sha(dest),state=state))
    (PREP/'decoded_cpu.json').write_text(json.dumps(dict(rows=rows,GPU=0),indent=2)+'\n')
    print(json.dumps(dict(videos=len(rows),frames=sum(len(r['samples']) for r in rows),GPU=0)))
if __name__=='__main__':main()
