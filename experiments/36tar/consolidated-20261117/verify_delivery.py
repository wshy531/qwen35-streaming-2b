import csv,hashlib,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'delivery_manifest.json').read_text(encoding='utf8'))
for rel,want in manifest.items():
 p=ROOT/rel;assert p.resolve().is_relative_to(ROOT.resolve())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==want,rel
 if p.suffix=='.py':compile(p.read_bytes(),rel,'exec')
for row in json.loads((ROOT/'source_manifest.json').read_text(encoding='utf8')):
 assert hashlib.sha256((ROOT/row['snapshot']).read_bytes()).hexdigest()==row['sha256']
rows=list(csv.DictReader((ROOT/'lessons/native-lora/outcomes.csv').open(encoding='utf8')))
assert len(rows)==9590 and len({(r['item'],r['method']) for r in rows})==9590
assert len({r['item'] for r in rows})==959 and len({r['source'] for r in rows})==192
summary=json.loads((ROOT/'lessons/native-lora/analysis.json').read_text())['table']
for method,expected in summary.items():
 selected=[r for r in rows if r['method']==method]
 assert len(selected)==expected['n']==959
 assert sum(int(r['correct']) for r in selected)==expected['content_correct']
 assert sum(int(r['unparsed']) for r in selected)==expected['unparsed']
print(json.dumps(dict(passed=True,files=len(manifest),questions=959,answers=9590,source_hashes=192)))
