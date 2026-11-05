"""Plan or fetch only fixed development videos from a completed pinned tar index."""
import argparse
import hashlib
import json
from pathlib import Path

from index_ovo_tar import HeaderReader, relative_name

REVISION='fec29e3385747b5642d995370143ba92d2819bd2'


def make_plan(index, manifest, count, budget):
    if index['status']!='complete' or index['revision']!=REVISION:
        raise ValueError('A completed pinned archive index is required')
    sources=[]
    for row in manifest:
        if row['dataset']=='ovo' and row['split']=='development' and row['source_id'] not in sources:
            sources.append(row['source_id'])
    selected=[]
    for source in sources[:count]:
        matches=[m for m in index['members'] if m['regular_file'] and m['relative_path']==source]
        if len(matches)!=1:
            raise ValueError('Missing or ambiguous archive member: '+source)
        member=matches[0]
        if relative_name(source)!=source or not 0<member['bytes']<=2*1024**3:
            raise ValueError('Unsafe media member or unexpected size')
        questions=[r['id'] for r in manifest if r['dataset']=='ovo' and r['source_id']==source]
        selected.append({**member,'question_ids':questions})
    if len(selected)!=count:
        raise ValueError('Not enough frozen development sources')
    total=sum(m['bytes'] for m in selected)
    return dict(status='planned_not_downloaded',revision=REVISION,
        selection='first development sources in pre-existing candidate order; no result-based selection',
        videos=selected,video_count=len(selected),bytes=total,budget_bytes=budget,
        fits_budget=total<=budget,
        integrity='Pinned HTTPS archive ranges and tar header checks; compute per-video SHA256 after download. Whole archive digest is not recomputed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--parts',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan-out',type=Path)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--budget-bytes',type=int,default=4*1024**3)
    p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--download-approved',action='store_true')
    args=p.parse_args()
    if not 1<=args.count<=8 or not 1<=args.seconds<=900 or not 0<args.budget_bytes<=8*1024**3:
        raise ValueError('Invalid bounded pilot budget')
    index=json.loads(args.index.read_text(encoding='utf-8'))
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    plan=make_plan(index,manifest,args.count,args.budget_bytes)
    if args.plan_out:
        with args.plan_out.open('x',encoding='utf-8') as f:
            f.write(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if not args.download_approved:
        return
    if not plan['fits_budget']:
        raise ValueError('Planned files exceed approved byte budget')
    output=args.out.resolve()
    output.mkdir(parents=True,exist_ok=False)
    parts=sorted((r for r in json.loads(args.parts.read_text()) if r['path'].startswith('src_videos.tar.part')),key=lambda r:r['path'])
    if len(parts)!=5 or sum(r['size'] for r in parts)!=index['archive_bytes']:
        raise ValueError('Archive partition identity mismatch')
    reader=HeaderReader(parts,REVISION,args.seconds,max_bytes=args.budget_bytes,
                        max_requests=2048,max_read=8*1024*1024)
    completed=[]
    try:
        for member in plan['videos']:
            target=output/member['relative_path']
            if not target.resolve().is_relative_to(output):
                raise ValueError('Media target escapes output')
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.part')
            digest=hashlib.sha256()
            with partial.open('xb') as f:
                for offset in range(0,member['bytes'],8*1024*1024):
                    size=min(8*1024*1024,member['bytes']-offset)
                    chunk=reader.read(member['data_offset']+offset,size)
                    digest.update(chunk);f.write(chunk)
            if partial.stat().st_size!=member['bytes']:
                raise ValueError('Video byte count mismatch')
            partial.rename(target)
            completed.append(dict(source=member['relative_path'],bytes=member['bytes'],
                sha256=digest.hexdigest(),path=member['relative_path'],review_status='not_decoded_or_reviewed'))
            (output/'download_manifest.json').write_text(json.dumps(dict(revision=REVISION,files=completed),indent=2)+'\n',encoding='utf-8')
            print(json.dumps(completed[-1]),flush=True)
    except Exception as error:
        (output/'download_error.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            completed_files=len(completed),bytes_received=reader.bytes,preserve_partials=True),indent=2)+'\n',encoding='utf-8')
        raise RuntimeError('Pilot fetch stopped: '+type(error).__name__) from None
    finally:
        reader.client.close()


if __name__=='__main__':
    main()
