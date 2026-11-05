"""One bounded preflight of the author's public NExTVideo ZIP; no bulk download."""
import argparse
from html.parser import HTMLParser
import json
import struct
import time
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import urlencode, urljoin, urlparse

import httpx

SOURCE = 'https://drive.google.com/uc?export=download&id=1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH'


def parse_directory(body):
    entries = []
    pos = 0
    names = set()
    while pos < len(body):
        values = struct.unpack_from('<4s6H3L5H2L', body, pos)
        if values[0] != b'PK\x01\x02':
            raise ValueError('Invalid ZIP central-directory signature')
        flags, method = values[3:5]
        crc, compressed, size = values[7:10]
        name_n, extra_n, comment_n = values[10:13]
        offset = values[16]
        name_bytes = body[pos+46:pos+46+name_n]
        extra = body[pos+46+name_n:pos+46+name_n+extra_n]
        name = name_bytes.decode('utf-8' if flags & 0x800 else 'cp437')
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or name in names:
            raise ValueError('Unsafe or duplicate archive member')
        names.add(name)
        at = 0
        while at+4 <= len(extra):
            kind, length = struct.unpack_from('<HH', extra, at)
            data = extra[at+4:at+4+length]
            if len(data) != length:
                raise ValueError('Truncated ZIP extra field')
            if kind == 1:
                index = 0
                for key in ('size', 'compressed', 'offset'):
                    old = {'size':size,'compressed':compressed,'offset':offset}[key]
                    if old == 0xffffffff:
                        value, = struct.unpack_from('<Q', data, index)
                        index += 8
                        if key == 'size':size = value
                        elif key == 'compressed':compressed = value
                        else:offset = value
            at += 4+length
        entries.append(dict(path=name, flags=flags, method=method, crc32=f'{crc:08x}',
                            compressed_bytes=compressed, bytes=size, local_header_offset=offset))
        pos += 46+name_n+extra_n+comment_n
    if pos != len(body):
        raise ValueError('Directory size mismatch')
    return entries


class PublicDownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.params = {}
        self.in_form = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs.get('action')
            self.in_form = True
        if tag == 'input' and self.in_form and attrs.get('name'):
            self.params[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.in_form = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--manifest', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic()+120
    report = dict(source=SOURCE, purpose='range_and_zip_directory_preflight_only',
                  request_limit=8, body_byte_limit=8*1024*1024, requests=[], bytes_read=0)
    url = SOURCE
    seen_form = False
    prior = None
    directory = None
    if a.resume_from:
        prior = json.loads((a.resume_from/'report.json').read_text(encoding='utf-8'))
        if prior['source'] != SOURCE or prior['status'] != 'range_supported_zip_tail_found':
            raise ValueError('Incompatible prior preflight')
        tail = (a.resume_from/'zip-tail.bin').read_bytes()
        marker = tail.rfind(b'PK\x06\x06')
        if marker < 0:
            raise ValueError('Expected ZIP64 directory identity')
        z64 = struct.unpack_from('<4sQ2H2L4Q', tail, marker)
        directory = dict(entries=z64[7], bytes=z64[8], offset=z64[9], archive_bytes=prior['archive_bytes'])
        if directory['bytes'] > 6*1024*1024 or directory['offset']+directory['bytes']>directory['archive_bytes']:
            raise ValueError('ZIP directory exceeds budget or archive bounds')
        report['requests'] = prior['requests'].copy()
        report['bytes_read'] = prior['bytes_read']
        report['resumed_from'] = str(a.resume_from)
        report['directory_identity'] = directory
    try:
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            for count in range(8-len(report['requests'])):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Preflight wall budget exceeded')
                host = urlparse(url).hostname or ''
                if not (host == 'drive.google.com' or host == 'drive.usercontent.google.com'
                        or host.endswith('.googleusercontent.com')):
                    raise ValueError('Download host outside Google public delivery')
                wanted_range = ('bytes='+str(directory['offset'])+'-'+str(directory['offset']+directory['bytes']-1)
                                if directory else 'bytes=-131072')
                with client.stream('GET', url, headers={'Range': wanted_range, 'Accept-Encoding':'identity'}) as r:
                    entry = dict(host=host, status=r.status_code, content_type=r.headers.get('Content-Type'),
                                 content_range=r.headers.get('Content-Range'), content_length=r.headers.get('Content-Length'))
                    report['requests'].append(entry)
                    if r.status_code in (301, 302, 303, 307, 308):
                        url = urljoin(url, r.headers['Location'])
                        continue
                    if r.status_code == 200 and 'text/html' in r.headers.get('Content-Type',''):
                        body = bytearray()
                        for chunk in r.iter_bytes(16384):
                            report['bytes_read'] += len(chunk)
                            body.extend(chunk)
                            if len(body)>131072 or report['bytes_read']>8*1024*1024:
                                raise ValueError('Public HTML preflight limit exceeded')
                        parser = PublicDownloadForm()
                        parser.feed(body.decode('utf-8', errors='replace'))
                        if not seen_form and parser.action:
                            # Normal public large-file confirmation. Parameters remain ephemeral.
                            url = parser.action+'?'+urlencode(parser.params)
                            seen_form = True
                            report['public_large_file_form_followed'] = True
                            continue
                        report['status'] = 'public_html_no_usable_range_download'
                        break
                    if r.status_code != 206:
                        report['status'] = 'range_not_supported_or_http_error'
                        report['unranged_binary_body_read'] = False
                        break
                    cr = r.headers.get('Content-Range','')
                    if not cr.startswith('bytes ') or '/' not in cr:
                        raise ValueError('Missing validated range bounds')
                    interval, total = cr[6:].split('/')
                    lo, hi = map(int, interval.split('-'))
                    total = int(total)
                    if not 0 <= lo <= hi < total or hi-lo+1 > 8*1024*1024:
                        raise ValueError('Invalid or oversized range')
                    body = bytearray()
                    for chunk in r.iter_bytes(16384):
                        body.extend(chunk)
                        report['bytes_read'] += len(chunk)
                        if len(body)>hi-lo+1 or report['bytes_read']>8*1024*1024 or time.monotonic()>=deadline:
                            raise ValueError('Range budget exceeded')
                    if len(body) != hi-lo+1:
                        raise ValueError('Truncated range response')
                    if directory:
                        if lo != directory['offset'] or hi+1 != directory['offset']+directory['bytes'] or total != directory['archive_bytes']:
                            raise ValueError('Directory response differs from validated tail')
                        entries = parse_directory(body)
                        if len(entries) != directory['entries']:
                            raise ValueError('Directory entry count mismatch')
                        (a.out/'central-directory.bin').write_bytes(body)
                        (a.out/'archive_index.json').write_text(json.dumps(dict(source=SOURCE, archive_bytes=total,
                            entries=entries), indent=2)+'\n', encoding='utf-8')
                        report['entries'] = len(entries)
                        report['status'] = 'zip_directory_validated'
                        if a.manifest:
                            candidates = json.loads(a.manifest.read_text(encoding='utf-8'))
                            wanted = {q['source_id']:q['source_path'] for q in candidates if q['dataset']=='nextgqa'}
                            matches = {}
                            for vid, mapped in wanted.items():
                                hits = [e for e in entries if e['path'].endswith('/'+str(mapped)+'.mp4')
                                        or e['path'] == str(mapped)+'.mp4' or e['path'].endswith('/'+str(mapped))]
                                if len(hits) == 1:
                                    matches[vid] = hits[0]
                            (a.out/'candidate_matches.json').write_text(json.dumps(matches, indent=2)+'\n', encoding='utf-8')
                            report['matched_candidate_sources'] = len(matches)
                            report['matched_compressed_bytes'] = sum(v['compressed_bytes'] for v in matches.values())
                        break
                    (a.out/'zip-tail.bin').write_bytes(body)
                    report['archive_bytes'] = total
                    report['tail_range'] = [lo,hi]
                    pos = body.rfind(b'PK\x05\x06')
                    if pos < 0:
                        report['status'] = 'range_supported_zip_directory_not_in_tail'
                        break
                    fields = struct.unpack_from('<4s4H2LH', body, pos)
                    report['eocd'] = dict(entries=fields[4], directory_bytes=fields[5], directory_offset=fields[6])
                    report['status'] = 'range_supported_zip_tail_found'
                    break
            else:
                report['status'] = 'request_budget_exhausted'
    except Exception as e:
        # Never log redirect URLs, confirmation parameters or exception text containing them.
        report['status'] = 'bounded_preflight_failed'
        report['error_type'] = type(e).__name__
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
