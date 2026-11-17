"""Fetch public metadata only; never fetch weights or submit a job."""
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/7b_transfer_prepare_20261107_01'
REPO='Qwen/Qwen2.5-VL-7B-Instruct'

def main():
    OUT.mkdir(exist_ok=True)
    url=f'https://huggingface.co/api/models/{REPO}?blobs=true'
    with urllib.request.urlopen(url,timeout=30) as response:
        raw=response.read(2*1024*1024)
    info=json.loads(raw)
    selected=[]
    for row in info['siblings']:
        name=row['rfilename']
        if name.endswith('.safetensors') or name in {
            'config.json','generation_config.json','preprocessor_config.json',
            'video_preprocessor_config.json','processor_config.json','chat_template.json',
            'chat_template.jinja','tokenizer.json','tokenizer_config.json','vocab.json',
            'merges.txt','special_tokens_map.json','added_tokens.json',
            'model.safetensors.index.json','README.md','LICENSE'}:
            selected.append(dict(name=name,size=row['size'],lfs_sha256=row.get('lfs',{}).get('sha256'),
                                 blob_id=row.get('blobId')))
    assert selected and all(r['lfs_sha256'] for r in selected if r['name'].endswith('.safetensors'))
    manifest=dict(repo=REPO,revision=info['sha'],files=selected,
                  total_bytes=sum(r['size'] for r in selected),weights_downloaded=False)
    (OUT/'official_metadata.json').write_bytes(raw)
    (OUT/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
