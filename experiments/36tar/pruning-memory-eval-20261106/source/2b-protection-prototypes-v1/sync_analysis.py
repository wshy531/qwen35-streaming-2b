"""Sync small derived analyses to the matching finalized owned run, never overwrite."""
import argparse,subprocess,shlex,json,hashlib,re,base64
from pathlib import Path
R=Path(__file__).resolve().parents[2]
REMOTE='/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'

def main(name):
    assert re.fullmatch(r'(isolated-recent|protect-stream|protect-channels)-[0-9]+',name)
    p=R/'runs'/name;done=p/'completed.json';assert json.loads(done.read_text(encoding='utf-8'))['status']=='complete'
    files=[p/n for n in ('analysis.json','lexical_replay.json','paired_changes.json') if (p/n).is_file()]
    assert files and sum(x.stat().st_size for x in files)<2**20
    manifest={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    check='''import json,sys,hashlib,base64
from pathlib import Path
r=Path(sys.argv[1]);p=r/'runs'/sys.argv[2];m=json.load(sys.stdin)
assert p.resolve().is_relative_to((r/'runs').resolve()) and not p.is_symlink()
assert (p/'completed.json').is_file()
for n,item in m.items():
 assert n in ('analysis.json','lexical_replay.json','paired_changes.json')
 data=base64.b64decode(item['data']);h=item['sha256'];assert hashlib.sha256(data).hexdigest()==h
 x=p/n
 assert not x.is_symlink()
 if not x.exists():
  with x.open('xb') as f:f.write(data)
 assert x.is_file() and hashlib.sha256(x.read_bytes()).hexdigest()==h,n
print(json.dumps(dict(verified=True,files=len(m))))
'''
    payload={x.name:dict(sha256=manifest[x.name],data=base64.b64encode(x.read_bytes()).decode('ascii')) for x in files}
    cmd=REMOTE+'/envs/isolated-recent-v1/bin/python -c '+shlex.quote(check)+' '+' '.join(shlex.quote(x) for x in (REMOTE,name))
    z=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ChaoSuan',cmd],input=json.dumps(payload),text=True,capture_output=True)
    assert z.returncode==0,z.stderr
    print(z.stdout)
    with (p/'analysis_synced.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');main(p.parse_args().name)
