"""Download pinned public Qwen assets using bounded buffers and official identities.

No model imports or GPU use. Weight download requires --include-weights and user approval.
The metadata must come from the official model revision API with blobs=true.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
PROCESSOR = ('chat_template.jinja','config.json','merges.txt','preprocessor_config.json',
             'tokenizer.json','tokenizer_config.json','video_preprocessor_config.json','vocab.json')
WEIGHTS = ('model.safetensors.index.json','model.safetensors-00001-of-00001.safetensors')


def inspect(path, item):
    size = item['size']
    digest = hashlib.sha256()
    blob = hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    count = 0
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            count += len(chunk)
            digest.update(chunk)
            blob.update(chunk)
    expected_lfs = item.get('lfs',{}).get('sha256')
    if count != size or (digest.hexdigest()!=expected_lfs if expected_lfs else blob.hexdigest()!=item['blobId']):
        raise ValueError('Official asset identity mismatch: '+path.name)
    return dict(name=path.name,bytes=count,sha256=digest.hexdigest(),expected_lfs_sha256=expected_lfs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-weights',action='store_true')
    parser.add_argument('--download-approved',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--range-bytes',type=int,default=0,
                        help='Use verified HTTP ranges for weights; 0 uses one request')
    parser.add_argument('--resume-range',action='store_true',
                        help='Explicitly resume a preserved weight .part; full digest is mandatory')
    args=parser.parse_args()
    if args.range_bytes and not 65536 <= args.range_bytes <= 64*1024*1024:
        raise ValueError('Range size must be between 64KiB and 64MiB')
    meta=json.loads(args.metadata.read_text(encoding='utf-8'))
    if meta['sha']!=REVISION or meta.get('id')!=MODEL:
        raise ValueError('Unexpected model identity')
    names=PROCESSOR+(WEIGHTS if args.include_weights else ())
    items={x['rfilename']:x for x in meta['siblings']}
    for name in names:
        item=items[name]
        if not 0<item['size']<=5*1024**3:
            raise ValueError('Unexpected asset size')
        if name.endswith('.safetensors') and not item.get('lfs',{}).get('sha256'):
            raise ValueError('Official weight digest required')
    total=sum(items[n]['size'] for n in names)
    if total>5*1024**3:
        raise ValueError('Download exceeds model budget')
    print(json.dumps(dict(model=MODEL,revision=REVISION,bytes=total,files=len(names),dry_run=args.dry_run)),flush=True)
    if args.dry_run:
        return
    if not args.download_approved:
        raise ValueError('Explicit download approval marker required')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    for manifest_name in ('PROCESSOR_MANIFEST.json','DOWNLOAD_MANIFEST.json'):
        if (output/manifest_name).exists():
            previous=json.loads((output/manifest_name).read_text(encoding='utf-8'))
            if (previous.get('model'),previous.get('revision'))!=(MODEL,REVISION):
                raise ValueError('Destination belongs to another model revision')
    deadline=time.monotonic()+1800
    records=[]
    for name in names:
        item=items[name]
        target=output/name
        if not target.exists():
            partial=output/(name+'.part')
            resume = partial.exists() and args.resume_range and args.range_bytes and name.endswith('.safetensors')
            if partial.exists() and not resume:
                raise ValueError('Partial download exists; inspect before resuming: '+name)
            start_offset=partial.stat().st_size if resume else 0
            if start_offset>item['size']:
                raise ValueError('Partial file exceeds official size')
            try:
                base_url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
                if args.range_bytes and name.endswith('.safetensors'):
                    with partial.open('ab' if resume else 'xb') as f:
                        for start in range(start_offset,item['size'],args.range_bytes):
                            end=min(start+args.range_bytes,item['size'])-1
                            request=urllib.request.Request(base_url+f'?download=true&offset={start}',
                                headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'})
                            with urllib.request.urlopen(request,timeout=30) as response:
                                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{item["size"]}':
                                    raise ValueError('Server returned an unexpected byte range')
                                count=0
                                while count < end-start+1:
                                    if time.monotonic()>deadline:
                                        raise TimeoutError('Total download time budget exceeded')
                                    chunk=response.read(min(1024*1024,end-start+1-count))
                                    if not chunk:
                                        raise ValueError('Truncated byte range')
                                    count+=len(chunk)
                                    f.write(chunk)
                            if start//args.range_bytes%4==0 or end+1==item['size']:
                                print(json.dumps(dict(file=name,downloaded_bytes=end+1,total_bytes=item['size'])),flush=True)
                else:
                    with urllib.request.urlopen(base_url,timeout=30) as response,partial.open('xb') as f:
                        count=0
                        while True:
                            if time.monotonic()>deadline:
                                raise TimeoutError('Total download time budget exceeded')
                            chunk=response.read(1024*1024)
                            if not chunk:
                                break
                            count+=len(chunk)
                            if count>item['size']:
                                raise ValueError('File exceeds official byte count')
                            f.write(chunk)
            except Exception as e:
                # Avoid signed CDN URLs in logs; preserve .part for inspection.
                raise RuntimeError(name+': '+type(e).__name__) from None
            record=inspect(partial,item)
            partial.rename(target)
            record['name']=name
        else:
            record=inspect(target,item)
        records.append(record)
        print(name,'verified',flush=True)
    manifest=output/('DOWNLOAD_MANIFEST.json' if args.include_weights else 'PROCESSOR_MANIFEST.json')
    manifest.write_text(json.dumps(dict(model=MODEL,revision=REVISION,files=records),indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
