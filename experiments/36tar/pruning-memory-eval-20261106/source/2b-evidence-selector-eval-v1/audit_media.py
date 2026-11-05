"""Read-only own-project media integrity and light header checks."""
import argparse
from pathlib import Path
from selector_qa_common import ROOT,PREP,read,save,sha,media_path

def main(out):
    import av
    records=[]
    for s in read(PREP/'plan.json')['sources']:
        p=media_path(s)
        if p.stat().st_size!=s['bytes'] or sha(p)!=s['sha256']:raise ValueError('Read-only media checksum mismatch: '+s['name'])
        with av.open(str(p)) as c:
            v=c.streams.video[0];v.codec_context.thread_count=1
            f=next(c.decode(v))
            if not v.time_base or f.time is None:raise ValueError('Real timestamps required')
            duration=float(v.duration*v.time_base) if v.duration is not None else c.duration/av.time_base
            if duration+1e-5<s['cutoff_s']:raise ValueError('Query prefix exceeds video duration')
            records.append(dict(source=s['name'],bytes=s['bytes'],sha256=s['sha256'],
                width=v.width,height=v.height,duration=duration,cutoff=s['cutoff_s']))
    save(out,dict(status='passed_own_media_SHA_and_headers',sources=records,
        partner_paths_read=sum('authorized_shared_readonly_path' in s for s in read(PREP/'plan.json')['sources']),
        GPU_jobs=0,full_visual_review=False))
    print('Passed',len(records),'read-only media checks; no GPU')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
