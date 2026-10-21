"""One bounded public-metadata pass; never downloads any video or uses a token."""
import hashlib,json,time
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/2b_memory_value_design_20261021_01'


def main():
    urls=dict(official_QA='https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/143fb31/EgoLifeQA/EgoLifeQA_A1_JAKE.json',
        day1_listing='https://huggingface.co/api/datasets/lmms-lab/EgoLife/tree/143fb31/A1_JAKE/DAY1?limit=1000')
    records={};total=0;start=time.monotonic()
    for name,url in urls.items():
        with urllib.request.urlopen(url,timeout=20) as response:
            data=bytearray()
            while True:
                block=response.read(65536)
                if not block:break
                data.extend(block);total+=len(block)
                assert len(data)<2*2**20 and total<4*2**20 and time.monotonic()-start<90
            parsed=json.loads(data);dest=OUT/(name+'.json')
            with dest.open('xb') as f:f.write(data)
            records[name]=dict(url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                full_repo_revision=response.headers.get('x-repo-commit'),
                has_next_page='rel="next"' in response.headers.get('Link',''),type=type(parsed).__name__,items=len(parsed))
    receipt=dict(status='public_metadata_only',records=records,total_bytes=total,video_bytes=0,
        no_credentials_used=True,no_retries=True,seconds=time.monotonic()-start,
        limitations='filenames/sizes and QA are not actual video PTS, visual admission, or continuity evidence')
    with (OUT/'metadata_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(receipt)


if __name__=='__main__':main()
