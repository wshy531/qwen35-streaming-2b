"""Standard-library integrity checks and raw-prediction rescoring; no model loads."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def raw(p):return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-root',type=Path);args=parser.parse_args()
    manifest=read(HERE/'PACKAGE_MANIFEST.json')
    for item in manifest['files']:
        p=HERE/item['path']
        assert p.resolve().is_relative_to(HERE.resolve())
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
    candidates=read(HERE/'catalog/candidate_pool385.json')
    key={r['id']:r for r in read(HERE/'catalog/evaluated171_catalog.json')}
    media=read(HERE/'catalog/media_manifest124.json')
    assert len(candidates)==len({r['id'] for r in candidates})==385
    assert len(key)==171 and len(media)==124
    spec=importlib.util.spec_from_file_location('delivery_scoring',HERE/'bundle/experiments/2b-short-event-v1/scoring.py')
    scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
    tables={}
    for name in ['natural-v1','short-event-v1']:
        folder=HERE/'evidence'/name/'result'
        predictions=(raw(folder/'native/predictions.jsonl')+raw(folder/'memory/predictions.jsonl')
                     if name=='natural-v1' else raw(folder/'main/predictions.jsonl'))
        assert len(predictions)==(1026 if name=='natural-v1' else 1197)
        assert len({(p['id'],p['arm']) for p in predictions})==len(predictions)
        counts=defaultdict(lambda:dict(questions=0,strict_correct=0,content_correct=0))
        for p in predictions:
            k=key[p['id']];v=scoring.measure(p['text'] if p['status']=='ok' else '',len(k['options']),k['answer_index'])
            dst=counts[k['cohort']+'/'+p['arm']];dst['questions']+=1
            dst['strict_correct']+=v['strict_correct'];dst['content_correct']+=v['content_correct']
        tables[name]=dict(counts)
    assert tables['natural-v1']['public74/native_original']['strict_correct']==49
    assert tables['short-event-v1']['new50/B02_cap4']['strict_correct']==27
    checked=0
    if args.dataset_root:
        root=args.dataset_root.resolve()
        for item in media:
            p=root/item['relative_path'];assert p.resolve().is_relative_to(root)
            assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['relative_path']
            checked+=1
    print(json.dumps(dict(status='passed',package_files=len(manifest['files']),candidate_questions=385,
                         evaluated_questions=171,media_manifest_files=124,actual_videos_checked=checked,
                         raw_scores=tables),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
