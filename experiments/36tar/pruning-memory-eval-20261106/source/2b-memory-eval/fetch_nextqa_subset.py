"""Fetch preselected development clips from the official ZIP with CRC verification."""
import argparse
import hashlib
import json
import re
import struct
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode, urljoin, urlparse
import zlib

import httpx
from probe_nextqa_zip import SOURCE, PublicDownloadForm


class DriveRanges:
    def __init__(self, archive_bytes, max_bytes=64*1024*1024, seconds=300):
        self.archive_bytes = archive_bytes
        self.max_bytes = max_bytes
        self.bytes_read = 0
        self.requests = 0
        self.deadline = time.monotonic()+seconds
        self.url = SOURCE
        self.last_response = None
        self.client = httpx.Client(timeout=25, follow_redirects=False)

    def chunks(self, lo, length):
        if not 0 <= lo < lo+length <= self.archive_bytes or length <= 0:
            raise ValueError('Invalid archive range')
        seen_form = False
        for _ in range(5):
            self.requests += 1
            if self.requests > 24 or time.monotonic() >= self.deadline:
                raise TimeoutError('Acquisition request/time budget exhausted')
            host = urlparse(self.url).hostname or ''
            if not (host in {'drive.google.com','drive.usercontent.google.com'} or host.endswith('.googleusercontent.com')):
                raise ValueError('Download host outside public Google delivery')
            with self.client.stream('GET', self.url, headers={'Range':f'bytes={lo}-{lo+length-1}',
                    'Accept-Encoding':'identity', 'Cache-Control':'no-cache'}) as r:
                self.last_response = dict(status=r.status_code, content_range=r.headers.get('Content-Range'),
                    content_type=r.headers.get('Content-Type'), expected_range=[lo,lo+length-1])
                if r.status_code in (301,302,303,307,308):
                    self.url = urljoin(self.url, r.headers['Location'])
                    continue
                if r.status_code == 200 and 'text/html' in r.headers.get('Content-Type','') and not seen_form:
                    body = bytearray()
                    for chunk in r.iter_bytes(8192):
                        self.bytes_read += len(chunk)
                        body.extend(chunk)
                        if len(body)>131072 or self.bytes_read>self.max_bytes:
                            raise ValueError('Public confirmation page exceeds budget')
                    form = PublicDownloadForm()
                    form.feed(body.decode('utf-8', errors='replace'))
                    if not form.action:
                        raise ValueError('No normal public download confirmation')
                    self.url = form.action+'?'+urlencode(form.params)
                    seen_form = True
                    continue
                if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {lo}-{lo+length-1}/{self.archive_bytes}':
                    raise ValueError('Server did not honor exact archive range')
                got = 0
                for chunk in r.iter_bytes(65536):
                    got += len(chunk)
                    self.bytes_read += len(chunk)
                    if got>length or self.bytes_read>self.max_bytes or time.monotonic()>=self.deadline:
                        raise TimeoutError('Media acquisition byte/time budget exhausted')
                    yield chunk
                if got != length:
                    raise ValueError('Truncated compressed payload')
                return
        raise ValueError('Public download redirect budget exhausted')


def make_plan(manifest, index, count):
    selected = list(dict.fromkeys(q['source_id'] for q in manifest
        if q['dataset']=='nextgqa' and q['split']=='development'))[:count]
    if len(selected) != count or index['source'] != SOURCE:
        raise ValueError('Expected fixed development sources and official source')
    plan = []
    for vid in selected:
        mapped = next(q['source_path'] for q in manifest if q['dataset']=='nextgqa' and q['source_id']==vid)
        matches = [x for x in index['entries'] if x['path']=='NExTVideo/'+mapped+'.mp4']
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous archive source')
        e = matches[0]
        if e['flags'] & 1 or e['method'] not in (0,8) or not 0<e['bytes']<=64*1024*1024:
            raise ValueError('Unsupported or oversized ZIP member')
        path = PurePosixPath(e['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in e['path']:
            raise ValueError('Unsafe output member')
        plan.append(dict(source_id=vid, **e))
    if sum(x['compressed_bytes']+4096 for x in plan)>60*1024*1024:
        raise ValueError('Pilot exceeds fixed network budget')
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=4,choices=range(1,5))
    p.add_argument('--download-approved',action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    index = json.loads(a.index.read_text(encoding='utf-8'))
    plan = make_plan(manifest, index, a.count)
    if not a.download_approved:
        print(json.dumps(dict(files=plan, compressed_bytes=sum(x['compressed_bytes'] for x in plan),
                              output_bytes=sum(x['bytes'] for x in plan)),indent=2))
        return
    a.out.mkdir(parents=True,exist_ok=False)
    reader = DriveRanges(index['archive_bytes'])
    receipt = dict(source=SOURCE, archive_bytes=index['archive_bytes'],
        index_sha256=hashlib.sha256(a.index.read_bytes()).hexdigest(), files=[], status='running')
    try:
        for e in plan:
            header = b''.join(reader.chunks(e['local_header_offset'],4096))
            h = struct.unpack_from('<4s5H3L2H',header)
            if h[0]!=b'PK\x03\x04' or h[2]!=e['flags'] or h[3]!=e['method']:
                raise ValueError('Local ZIP header conflicts with central directory')
            name_n, extra_n = h[-2:]
            if 30+name_n+extra_n>len(header):
                raise ValueError('ZIP local header exceeds bounded read')
            name = header[30:30+name_n].decode('utf-8' if e['flags']&0x800 else 'cp437')
            if name != e['path']:
                raise ValueError('ZIP member identity mismatch')
            start = e['local_header_offset']+30+name_n+extra_n
            output = a.out/e['path']
            output.parent.mkdir(parents=True,exist_ok=True)
            partial = output.with_suffix('.mp4.partial')
            decoder = zlib.decompressobj(-15) if e['method']==8 else None
            size = crc = 0
            sha = hashlib.sha256()
            with partial.open('xb') as f:
                def write(data):
                    nonlocal size,crc
                    size += len(data)
                    if size>e['bytes']:
                        raise ValueError('Decompressed member exceeds declared size')
                    crc = zlib.crc32(data,crc)
                    sha.update(data)
                    f.write(data)
                for chunk in reader.chunks(start,e['compressed_bytes']):
                    if decoder:
                        data = decoder.decompress(chunk, e['bytes']-size+1)
                        if decoder.unconsumed_tail:
                            raise ValueError('Decompression exceeded declared member size')
                        write(data)
                    else:
                        write(chunk)
                if decoder:
                    write(decoder.flush())
                    if not decoder.eof or decoder.unused_data:
                        raise ValueError('Invalid compressed member termination')
            if size!=e['bytes'] or f'{crc & 0xffffffff:08x}'!=e['crc32']:
                raise ValueError('ZIP length or CRC mismatch')
            partial.rename(output)
            receipt['files'].append(dict(**e,sha256=sha.hexdigest(),crc_verified=True))
            (a.out/'download_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        receipt['status'] = 'complete'
    except Exception as err:
        receipt['status'] = 'failed_partial_preserved'
        receipt['error_type'] = type(err).__name__
        receipt['last_response'] = reader.last_response
        if isinstance(err, ValueError):
            receipt['failure_reason'] = str(err)
    finally:
        reader.client.close()
        receipt.update(bytes_read=reader.bytes_read, requests=reader.requests)
        (a.out/'download_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],files=len(receipt['files']),bytes_read=reader.bytes_read,
                         requests=reader.requests,error_type=receipt.get('error_type'),
                         failure_reason=receipt.get('failure_reason'),last_response=receipt.get('last_response')),indent=2))
    if receipt['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
