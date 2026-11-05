"""Bounded mirror metadata check only. Never fetches video contents."""
import hashlib,json,re,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/delayed_training_candidates_20261104_01'
REPO='jinyoungkim/NExT-GQA'
REV='5d1c262802efe7e5331629ca01e67916867590a5'


def main():
    sources=json.loads((OUT/'sources.json').read_text(encoding='utf-8'))
    assert len(sources)==150 and sum(s['bytes'] for s in sources)==858086651
    marker=OUT/'metadata_probe_started.json'
    with marker.open('x',encoding='utf-8') as f:json.dump(dict(attempts=1,media_downloads=0,byte_cap=2*2**20),f)
    started=time.monotonic();body=bytearray()
    try:
        with httpx.Client(timeout=httpx.Timeout(30,connect=10),follow_redirects=True) as client:
            with client.stream('POST',f'https://huggingface.co/api/datasets/{REPO}/paths-info/{REV}',
                               json={'paths':[s['archive_path'] for s in sources],'expand':False}) as r:
                r.raise_for_status()
                for block in r.iter_bytes(65536):
                    body.extend(block)
                    if len(body)>2*2**20 or time.monotonic()-started>90:raise TimeoutError('metadata limit')
        rows=json.loads(body);index={r['path']:r for r in rows}
        assert len(rows)==len(index)==150 and set(index)=={s['archive_path'] for s in sources}
        files=[]
        for s in sources:
            r=index[s['archive_path']];digest=r['lfs']['oid']
            assert r['size']==s['bytes'] and re.fullmatch('[a-f0-9]{64}',digest)
            files.append(dict(s,mirror_path=s['archive_path'],mirror_lfs_sha256=digest,official_zip_crc32=s['crc32']))
        result=dict(repository=REPO,revision=REV,files=files,total_bytes=858086651,
            media_downloaded=0,metadata_bytes=len(body),seconds=time.monotonic()-started,
            sources_sha256=hashlib.sha256((OUT/'sources.json').read_bytes()).hexdigest(),
            status='metadata_verified_media_not_downloaded',download_authorization_required=True)
        with (OUT/'acquisition_plan.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    except Exception as error:
        (OUT/'metadata_probe_error.json').write_text(json.dumps(dict(error_type=type(error).__name__,
             metadata_bytes=len(body),seconds=time.monotonic()-started,media_downloaded=0,no_automatic_retry=True)),encoding='utf-8')
        raise RuntimeError('metadata probe stopped; receipt saved') from None


if __name__=='__main__':main()
