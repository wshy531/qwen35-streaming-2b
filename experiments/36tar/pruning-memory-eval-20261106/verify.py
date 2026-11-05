"""Offline integrity and anonymized outcome verification; standard library only."""
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    inventory = read('publication_manifest.json')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    assert actual == set(inventory) | {'publication_manifest.json'}
    for name, expected in inventory.items():
        p = ROOT/name
        assert p.is_file() and not p.is_symlink()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=name)
    manifest = read('source_manifest.json')
    for r in manifest:
        assert inventory[r['snapshot']] == r['sha256']
    with (ROOT/'results/outcomes.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 718
    assert len({(r['suite'],r['question'],r['method']) for r in rows}) == len(rows)
    assert Counter(r['job'] for r in rows) == {'220559':450, '220558':180, '220931':88}
    groups = defaultdict(lambda:dict(correct=0,total=0,unresolved=0))
    for r in rows:
        assert r['correct'] in ('0','1','') and r['unresolved'] in ('0','1')
        assert (r['correct']=='') == (r['unresolved']=='1')
        v = groups[(r['suite'],r['method'])]
        v['correct'] += r['correct']=='1'; v['total'] += 1; v['unresolved'] += int(r['unresolved'])
    derived = [dict(suite=k[0],method=k[1],**v) for k,v in sorted(groups.items())]
    assert derived == read('results/summary.json')
    assert groups['StreamingBench75','guarded_LS']['correct'] == 49
    assert groups['StreamingBench75','initial_LS']['correct'] == 50
    assert groups['paired_draft22','combined']['correct'] == 19
    catalog = read('catalog/suite_index.json')
    for v in catalog.values(): assert v['count'] == len(v['records'])
    assert catalog['development']['count'] == 228 and catalog['regression']['count'] == 153
    assert {r['id'] for r in catalog['regression']['records']} <= {r['id'] for r in catalog['development']['records']}
    assert catalog['short_validation']['count'] == 60
    print(json.dumps(dict(status='passed', files=len(inventory)+1, source_files=len(manifest), outcomes=len(rows), method_groups=len(groups), GPU=False, network=False)))

if __name__ == '__main__': main()
