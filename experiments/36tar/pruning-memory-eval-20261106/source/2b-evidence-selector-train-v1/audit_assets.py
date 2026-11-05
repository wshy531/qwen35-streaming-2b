"""Bounded read-only asset audit; no decoding or ML imports."""
import time,json
from prepare_data import ROOT,PREP,read,save,sha

def main():
    start=time.monotonic();short=read(PREP/'short_sources.json');long=read(PREP/'long_sources.json');checked=[]
    for s in short:
        p=ROOT/'datasets/VideoCon-support-v1/videos'/(s['video_id']+'.mp4')
        if p.stat().st_size!=s['media_bytes'] or sha(p)!=s['media_sha256']:raise ValueError('short asset mismatch')
        checked.append(dict(kind='short',source=s['id'],bytes=p.stat().st_size,sha256=s['media_sha256']))
        if time.monotonic()-start>100:raise TimeoutError('bounded asset hash audit')
    import av
    for s in long:
        p=ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23'/s['relative_path']
        if p.stat().st_size!=s['bytes'] or sha(p)!=s['sha256']:raise ValueError('long asset mismatch')
        with av.open(str(p)) as c:
            v=c.streams.video[0];fps=str(v.average_rate)
            if fps!='12':raise ValueError('registered one-FPS prefix count requires audited 12FPS assets')
        checked.append(dict(kind='long',source=s['video_id'],bytes=s['bytes'],sha256=s['sha256'],fps=fps))
        if time.monotonic()-start>100:raise TimeoutError('bounded asset hash audit')
    save(PREP/'cluster_asset_audit.json',dict(status='passed_SHA_and_headers',sources=checked,seconds=time.monotonic()-start,
        GPU_jobs=0,decoding=False,downloads=0,partner_access=False))
    print(dict(sources=len(checked),seconds=time.monotonic()-start,downloads=0))
if __name__=='__main__':main()
