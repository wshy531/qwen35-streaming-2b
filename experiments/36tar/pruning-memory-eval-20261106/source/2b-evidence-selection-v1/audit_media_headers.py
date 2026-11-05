"""One header per frozen file; bounded CPU/IO, no video decoding or GPU."""
from pathlib import Path
import argparse,time
from prepare import ROOT,PREP,read,save,sha

def main(media,output):
    import av
    allowed=[ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos',
        Path('/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos')]
    if media.resolve() not in [p.resolve() for p in allowed]:raise ValueError('only authorized exact media directory')
    start=time.monotonic();rows=[]
    if output.exists():raise FileExistsError(output)
    for s in read(PREP/'runtime_plan.json')['sources']:
        p=media/s['name']
        if p.resolve().parent!=media.resolve() or p.stat().st_size!=s['bytes'] or sha(p)!=s['sha256']:
            raise ValueError('frozen media size/SHA mismatch')
        with av.open(str(p)) as c:
            v=c.streams.video[0]
            duration=float(v.duration*v.time_base) if v.duration is not None else c.duration/1e6
            if duration<s['cutoff_s'] or duration>1300:raise ValueError('media duration outside frozen prefix')
            rows.append(dict(name=s['name'],bytes=s['bytes'],sha256=s['sha256'],
                duration_s=duration,width=v.width,height=v.height,average_rate=str(v.average_rate),
                frozen_cutoff_s=s['cutoff_s'],complete_pairs=s['max_pairs']))
        if time.monotonic()-start>100:raise TimeoutError('bounded CPU header/hash window exceeded')
    save(output,dict(status='passed_header_and_media_SHA_only',sources=rows,wall_seconds=time.monotonic()-start,
        no_decoding=True,pixel_verification='reuse old recent traces; recheck actual sampled pixels in GPU worker',GPU_jobs=0))
    print(dict(sources=len(rows),media_bytes=sum(r['bytes'] for r in rows),seconds=time.monotonic()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--media-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.media_root,a.output)
