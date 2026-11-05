"""One bounded author metadata fetch (<=10MiB), no videos/weights/credentials."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct-author-file',action='store_true',
        help='Use published author file once if anonymous commit API is rate limited; pin archived content SHA')
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[2]/'runs/development_audit_20261017_02'
    destination = out/'COIN.json'
    if destination.exists():
        raise FileExistsError('Existing metadata is not overwritten')
    start = time.monotonic()
    with httpx.Client(timeout=30,follow_redirects=True,
                      headers={'User-Agent':'qwen35-memory-eval-metadata-audit'}) as client:
        revision = None
        if not args.direct_author_file:
            commit = client.get('https://api.github.com/repos/coin-dataset/annotations/commits/master')
            commit.raise_for_status()
            revision = commit.json()['sha']
        url = f'https://raw.githubusercontent.com/coin-dataset/annotations/{revision or "master"}/COIN.json'
        content = bytearray()
        with client.stream('GET',url) as response:
            response.raise_for_status()
            if int(response.headers.get('content-length','0')) > 10*1024*1024:
                raise ValueError('Metadata cap exceeded')
            for chunk in response.iter_bytes(65536):
                if len(content)+len(chunk)>10*1024*1024 or time.monotonic()-start>90:
                    raise TimeoutError('Metadata budget exceeded')
                content.extend(chunk)
    data = json.loads(content)
    record = data['database']['_ymPcF51eHg']
    receipt = dict(source_url=url,revision=revision,bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        provenance='author_file_archived_by_content_hash' if revision is None else 'author_commit_and_content_hash',
        purpose='author_action_time_metadata_no_media_or_model')
    destination.write_bytes(content)
    (out/'coin_selected_metadata.json').write_text(json.dumps(
        {'_ymPcF51eHg':record},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'coin_fetch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(receipt=receipt,selected=record),ensure_ascii=False))


if __name__ == '__main__':
    main()
