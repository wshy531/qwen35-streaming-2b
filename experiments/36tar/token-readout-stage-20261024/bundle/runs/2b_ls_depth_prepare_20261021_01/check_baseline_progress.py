"""Read-only drift check, never reads truth or selects questions."""
import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import command


def main():
    code="""import json,pathlib
p=pathlib.Path.cwd().resolve();assert p.is_relative_to(pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
f=p/'runs/2b-ls-depth-217894/answers.jsonl';result=[]
if f.exists():
 with f.open() as stream:
  for line in stream:
   if not line.endswith(chr(10)):continue
   row=json.loads(line)
   if row['arm']!='LS_depth':result.append({k:row[k] for k in ['id','arm','generated_ids','input_ids_sha256','node_digest']})
print(json.dumps(result))
"""
    observed=json.loads(command(code,timeout=45))
    refs={(r['id'],r['arm']):r for r in json.loads((HERE/'historical_references_LOCAL_ONLY.json').read_text(encoding='utf-8'))['records']}
    checked=[];mismatches={key:[] for key in ('generated_ids','input_ids_sha256','node_digest')}
    for row in observed:
        key=row['id'],row['arm']
        if key not in refs:continue
        checked.append(key)
        for name in mismatches:
            if refs[key].get(name) and row[name]!=refs[key][name]:mismatches[name].append(dict(id=key[0],arm=key[1]))
    receipt=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checked=len(checked),mismatches=mismatches,
        no_ground_truth_read=True,no_source_or_protocol_change=True)
    with (HERE/'baseline_progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(receipt)+'\n')
    print(receipt)


if __name__=='__main__':main()
