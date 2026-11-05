"""Index a pinned split tar using directory headers only, without video payloads.

Signed redirect URLs remain only in memory. No media extraction is performed.
"""
import argparse
import bisect
import json
import tarfile
import time
from pathlib import Path, PurePosixPath

import httpx


class BudgetStop(RuntimeError):
    pass


class HeaderReader:
    def __init__(self, parts, revision, seconds=900, max_bytes=2*1024*1024,
                 max_requests=1500, max_read=1024*1024):
        self.parts=parts
        self.revision=revision
        self.starts=[]
        self.total=0
        for part in parts:
            self.starts.append(self.total)
            self.total+=part['size']
        self.deadline=time.monotonic()+seconds
        self.requests=0
        self.bytes=0
        self.max_bytes=max_bytes
        self.max_requests=max_requests
        self.max_read=max_read
        self.urls={}
        self.no_url_cache=set()
        self.client=httpx.Client(follow_redirects=True,timeout=httpx.Timeout(15,connect=10),
                                 headers={'Accept-Encoding':'identity'})

    def fetch_part(self, number, offset, length):
        if time.monotonic()>self.deadline or self.requests>=self.max_requests or self.bytes+length>self.max_bytes:
            raise BudgetStop('Header budget reached')
        part=self.parts[number]
        original=f'https://huggingface.co/datasets/JoeLeelyf/OVO-Bench/resolve/{self.revision}/{part["path"]}?download=true&offset={offset}'
        cached=self.urls.get(number) if number not in self.no_url_cache else None
        for attempt in range(2):
            self.requests+=1
            url=cached if cached and attempt==0 else original
            with self.client.stream('GET',url,headers={'Range':f'bytes={offset}-{offset+length-1}',
                                                      'Cache-Control':'no-cache'}) as response:
                if response.status_code==403 and cached and attempt==0:
                    self.no_url_cache.add(number)
                    continue
                expected=f'bytes {offset}-{offset+length-1}/{part["size"]}'
                if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                    raise ValueError('Unexpected status or Content-Range')
                data=bytearray()
                for chunk in response.iter_raw():
                    data.extend(chunk)
                    if len(data)>length:
                        raise ValueError('Header response exceeds requested byte count')
                if len(data)!=length:
                    raise ValueError('Truncated header response')
                self.bytes+=len(data)
                self.urls[number]=str(response.url)
                return bytes(data)
        raise ValueError('Unable to fetch header')

    def read(self, offset, length):
        if offset<0 or length<0 or offset+length>self.total or length>self.max_read:
            raise ValueError('Invalid bounded metadata read')
        data=bytearray()
        while len(data)<length:
            current=offset+len(data)
            number=bisect.bisect_right(self.starts,current)-1
            local=current-self.starts[number]
            count=min(length-len(data),self.parts[number]['size']-local)
            data.extend(self.fetch_part(number,local,count))
        return bytes(data)


def parse_pax(data):
    result={}
    position=0
    while position<len(data):
        space=data.find(b' ',position)
        if space<0:
            raise ValueError('Invalid PAX length')
        size=int(data[position:space])
        if size<=space-position+1 or position+size>len(data):
            raise ValueError('Invalid PAX record')
        record=data[space+1:position+size]
        if not record.endswith(b'\n'):
            raise ValueError('Invalid PAX terminator')
        key,separator,value=record[:-1].partition(b'=')
        if not separator:
            raise ValueError('Invalid PAX value')
        result[key.decode('utf-8')]=value.decode('utf-8')
        position+=size
    return result


def relative_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe archive name')
    parts=p.parts
    if parts and parts[0]=='src_videos':
        parts=parts[1:]
    return '/'.join(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--resume-from',type=Path,
                        help='Resume a plain-header checkpoint with no PAX/GNU extension state')
    args=parser.parse_args()
    if not 1<=args.seconds<=900 or len(args.revision)!=40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Invalid revision or time budget')
    parts=sorted((r for r in json.loads(args.parts.read_text()) if r['path'].startswith('src_videos.tar.part')),key=lambda r:r['path'])
    if len(parts)!=5:
        raise ValueError('Expected five official source parts')
    args.out.mkdir(parents=True,exist_ok=False)
    reader=HeaderReader(parts,args.revision,args.seconds)
    members=[]
    offset=0
    global_pax={}
    local_pax={}
    long_name=None
    status='in_progress'
    error_type=None
    if args.resume_from:
        previous=json.loads(args.resume_from.read_text(encoding='utf-8'))
        if previous['revision']!=args.revision or previous['archive_bytes']!=reader.total or previous['status']=='complete':
            raise ValueError('Resume identity mismatch')
        if previous['metadata_bytes']!=512*len(previous['members']):
            raise ValueError('Checkpoint extension state cannot be safely reconstructed')
        members=previous['members']
        offset=previous['next_header_offset']
        if members:
            last=members[-1]
            expected=last['header_offset']+512+((last['bytes']+511)//512)*512
            if offset!=expected:
                raise ValueError('Invalid checkpoint continuation offset')
        reader.bytes=previous['metadata_bytes']
        reader.requests=previous['requests']

    def save():
        manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
        selected={row['source_id'] for row in manifest if row['dataset']=='ovo'}
        found=[m for m in members if m['regular_file'] and m['relative_path'] in selected]
        result=dict(status=status,error_type=error_type,revision=args.revision,
            headers=len(members),metadata_bytes=reader.bytes,requests=reader.requests,
            next_header_offset=offset,archive_bytes=reader.total,no_video_payload_downloaded=True,
            matched_selected_sources=len({m['relative_path'] for m in found}),
            matched_selected_bytes=sum(m['bytes'] for m in found),
            selected_missing=sorted(selected-{m['relative_path'] for m in found}),
            matches=found,members=members,
            limitation='Directory checksums and HTTPS checked; whole archive LFS digest not recomputed.')
        (args.out/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in {'members','matches','selected_missing'}},ensure_ascii=False),flush=True)

    try:
        while offset+512<=reader.total:
            block=reader.read(offset,512)
            if block==b'\0'*512:
                status='complete'
                break
            member=tarfile.TarInfo.frombuf(block,'utf-8','surrogateescape')
            if member.type in {tarfile.XHDTYPE,tarfile.XGLTYPE,tarfile.GNUTYPE_LONGNAME,tarfile.GNUTYPE_LONGLINK}:
                payload=reader.read(offset+512,member.size)
                if member.type==tarfile.XHDTYPE:
                    local_pax.update(parse_pax(payload))
                elif member.type==tarfile.XGLTYPE:
                    global_pax.update(parse_pax(payload))
                elif member.type==tarfile.GNUTYPE_LONGNAME:
                    long_name=payload.rstrip(b'\0').decode('utf-8')
                offset+=512+((member.size+511)//512)*512
                continue
            pax={**global_pax,**local_pax}
            name=pax.get('path',long_name or member.name)
            size=int(pax.get('size',member.size))
            if size<0 or offset+512+size>reader.total:
                raise ValueError('Archive member exceeds archive size')
            members.append(dict(member_name=name,relative_path=relative_name(name),
                header_offset=offset,data_offset=offset+512,bytes=size,regular_file=member.isfile()))
            offset+=512+((size+511)//512)*512
            local_pax={};long_name=None
            if len(members)%50==0:
                save()
        if status!='complete':
            raise ValueError('Archive terminator not reached')
    except Exception as error:
        status='budget_stopped' if isinstance(error,BudgetStop) else 'stopped_on_error'
        error_type=type(error).__name__
    finally:
        reader.client.close()
        save()
    if status!='complete':
        raise SystemExit(2)


if __name__=='__main__':
    main()
