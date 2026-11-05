"""Extract fixed comparison/holdout members through bounded official TAR ranges."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from index_ovo_tar import HeaderReader,relative_name

REVISION='fec29e3385747b5642d995370143ba92d2819bd2'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--download-approved',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    prior=root/'runs/data_acquisition_preflight_20261016_01'
    index=json.loads((prior/'tar_index_resume01/index.json').read_text())
    if index['status']!='complete' or index['revision']!=REVISION:raise ValueError('pinned complete index required')
    sources=list(dict.fromkeys(r['source_path'] for r in rows if r['dataset']=='ovo' and r['split']!='development'))
    selected=[]
    for source in sources:
        found=[v for v in index['members'] if v['regular_file'] and v['relative_path']==source]
        if len(found)!=1 or relative_name(source)!=source:raise ValueError('ambiguous or unsafe member')
        qs=[r for r in rows if r['dataset']=='ovo' and r['source_path']==source]
        selected.append(dict(found[0],split=qs[0]['split'],question_ids=[q['id'] for q in qs]))
    total=sum(v['bytes'] for v in selected)
    budget=1536*1024*1024
    if len(selected)!=32 or total+512*len(selected)>budget:raise ValueError('approved count/byte budget exceeded')
    plan=dict(status='planned_not_downloaded',revision=REVISION,videos=selected,total_bytes=total,
        index_sha256=hashlib.sha256((prior/'tar_index_resume01/index.json').read_bytes()).hexdigest(),
        approved_budget_bytes=budget,seconds=2700,
        source='official OVO dataset pinned archive ranges; not full archive verification')
    args.plan.parent.mkdir(parents=True,exist_ok=True)
    with args.plan.open('x',encoding='utf-8') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(files=32,bytes=total,download=args.download_approved)),flush=True)
    if not args.download_approved:return
    out=args.out.resolve()
    if not out.is_relative_to((root/'datasets/2b-memory-eval').resolve()):raise ValueError('destination outside project data')
    out.mkdir(parents=True,exist_ok=False)
    parts=sorted((v for v in json.loads((prior/'ovo_file_tree.json').read_text()) if v['path'].startswith('src_videos.tar.part')),key=lambda v:v['path'])
    if len(parts)!=5 or sum(v['size'] for v in parts)!=index['archive_bytes']:raise ValueError('archive partition identity')
    reader=HeaderReader(parts,REVISION,2700,max_bytes=budget,max_requests=1000,max_read=8*1024*1024)
    done=[];entry=None;partial=None
    try:
        for entry in selected:
            target=out/entry['relative_path']
            if not target.resolve().is_relative_to(out):raise ValueError('unsafe target')
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.part');h=hashlib.sha256()
            with partial.open('xb') as f:
                for offset in range(0,entry['bytes'],8*1024*1024):
                    chunk=reader.read(entry['data_offset']+offset,min(8*1024*1024,entry['bytes']-offset))
                    h.update(chunk);f.write(chunk)
                    del chunk
            if partial.stat().st_size!=entry['bytes']:raise ValueError('size mismatch')
            partial.rename(target);partial=None
            done.append(dict(source_path=entry['relative_path'],local_path=str(target),bytes=entry['bytes'],
                sha256=h.hexdigest(),split=entry['split'],question_ids=entry['question_ids'],
                integrity='exact pinned official archive ranges; local size/SHA recorded, no upstream per-member SHA'))
            (out/'download_manifest.json').write_text(json.dumps(dict(status='downloaded_not_decoded_or_reviewed',
                revision=REVISION,files=done),indent=2)+'\n',encoding='utf-8')
            print(json.dumps(dict(source=entry['relative_path'],completed=len(done),bytes=entry['bytes'])),flush=True)
    except Exception as error:
        (out/'download_error.json').write_text(json.dumps(dict(status='stopped_no_automatic_retry',
            error_type=type(error).__name__,source_path=entry['relative_path'] if entry else None,
            completed=len(done),bytes_received=reader.bytes,partial_file=str(partial) if partial else None),indent=2)+'\n',encoding='utf-8')
        raise RuntimeError('bounded extraction stopped; see sanitized receipt') from None
    finally:reader.client.close()
    print(json.dumps(dict(completed=len(done),media_bytes=total,range_bytes=reader.bytes,requests=reader.requests)))


if __name__=='__main__':main()
