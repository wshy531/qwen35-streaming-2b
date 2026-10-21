"""Bounded official code inspection for timestamp representation, no execution."""
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/2b_memory_value_prepare_20261021_01'
def main():
    url='https://api.github.com/repos/EvolvingLMMs-Lab/EgoLife/git/trees/main?recursive=1';data=bytearray();start=time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'streaming-research-metadata'}),timeout=20) as r:
        while True:
            b=r.read(65536)
            if not b:break
            data.extend(b);assert len(data)<2**20 and time.monotonic()-start<30
    value=json.loads(data);assert not value.get('truncated')
    with (OUT/'author_repo_tree.json').open('xb') as f:f.write(data)
    print(dict(revision=value['sha'],source_SHA=hashlib.sha256(data).hexdigest(),
        candidate_files=[t['path'] for t in value['tree'] if t['path'].endswith(('.py','.md')) and 'EgoRAG' in t['path']],video_bytes=0))
if __name__=='__main__':main()
