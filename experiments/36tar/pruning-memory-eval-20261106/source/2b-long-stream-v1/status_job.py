"""Read only this authorized job's queue and compact project log counters."""
import json
import sys

from inventory_local import ROOT,OUT
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import command


def main():
    code="""import datetime,pathlib,json,subprocess
p=pathlib.Path.cwd();assert p.resolve().is_relative_to(pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve())
base=p/'runs/2b-long-stream-217776'
q=subprocess.run(['squeue','-j','217776','-h','-o','%i %T %M %R'],capture_output=True,text=True,timeout=15)
report=dict(queue=q.stdout,stages={},UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
for name in ['native','memory']:
 d=base/name;s={}
 for n in ['loaded.json','completed.json']:
  if (d/n).exists():s[n]={k:v for k,v in json.loads((d/n).read_text()).items() if k!='timelines'}
 if (d/'events.jsonl').exists():
  count={};last=None;ends={}
  with (d/'events.jsonl').open() as f:
   for line in f:
    try:item=json.loads(line)
    except json.JSONDecodeError:continue
    arm=item['arm'];e=item['event']
    if e['kind']=='query':count[arm]=count.get(arm,0)+1
    if e['kind']=='end':ends[arm]=ends.get(arm,0)+1
    last=dict(arm=arm,source=item['source_path'],event=e)
    if 'state' in item:last['state']={k:v for k,v in item['state'].items() if k in ['pairs','resets','main_seen_tokens','memory_from_pts','main_cache_bytes']}
  s.update(answers=count,finished_sources=ends,last=last)
 report['stages'][name]=s
err=p/'runs/2b-long-stream-217776.err'
if err.exists():
 text=err.read_text();report['error_tail']=text[-2500:] if 'Traceback' in text or 'Error' in text else 'no exception in stderr (known fallback/weight-load notices omitted)'
print(json.dumps(report))
"""
    result=command(code,timeout=35)
    (OUT/'latest_job_status.json').write_text(result,encoding='utf-8')
    print(result)


if __name__=='__main__':main()
