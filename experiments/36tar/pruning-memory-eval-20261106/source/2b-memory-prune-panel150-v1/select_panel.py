"""Select 75 questions per dataset by type and observed-prefix band."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'runs/ls_prune_expansion_prepare_20261103_01/regression300.json'
OUT = ROOT/'runs/memory_prune_panel150_prepare_20261103_01'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rank(text):
    return hashlib.sha256(('memory-prune-panel150-v1:'+text).encode()).hexdigest()


def band(q):
    t = q['query_time_s']
    edges = (15,) if q['dataset']=='NExT-QA' else (180,600)
    return sum(t>=edge for edge in edges)


def main():
    rows = read(SOURCE)
    selected = []
    for dataset in ('NExT-QA','StreamingBench'):
        strata = defaultdict(list)
        for q in rows:
            if q['dataset']==dataset:
                strata[q['type'],band(q)].append(q)
        assert sum(map(len,strata.values()))==150
        odd = sorted((k for k in strata if len(strata[k])%2),key=lambda k:rank(repr(k)))
        extra = set(odd[:len(odd)//2])
        for key, items in sorted(strata.items()):
            take = len(items)//2+int(key in extra)
            selected.extend(sorted(items,key=lambda q:rank(q['id']))[:take])
    assert len({q['id'] for q in selected})==150
    assert Counter(q['dataset'] for q in selected)=={'NExT-QA':75,'StreamingBench':75}
    audit = {r['video_id']:r for r in read(ROOT/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
    ref = read(ROOT/'runs/protection_streamingbench_prepare_20261103_01/causal_reference.json')['prefixes']
    pairs = {}
    for q in selected:
        ds, vid = q['dataset'],q['source_id']
        n = (len(audit[vid]['samples'])+1)//2 if ds=='NExT-QA' else ref[q['id']]['state']['complete_pairs']
        pairs[ds,vid] = max(pairs.get((ds,vid),0), n)
    total = sum(pairs.values())
    report = dict(questions=150, main_answers=2400, capacity_answers=216, main_pairs=total,
        vision_calls=4*total+429*3, media_files=len(pairs)+5,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        by_dataset={ds:dict(questions=75, sources=sum(k[0]==ds for k in pairs),
           pairs=sum(n for k,n in pairs.items() if k[0]==ds),
           types=dict(Counter(q['type'] for q in selected if q['dataset']==ds)),
           prefix_bands=dict(Counter(band(q) for q in selected if q['dataset']==ds)))
           for ds in ('NExT-QA','StreamingBench')})
    OUT.mkdir(exist_ok=True)
    for name, data in [('regression150.json',selected),('selection.json',report)]:
        with (OUT/name).open('x',encoding='utf-8') as f:
            json.dump(data,f,indent=2,ensure_ascii=False)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
