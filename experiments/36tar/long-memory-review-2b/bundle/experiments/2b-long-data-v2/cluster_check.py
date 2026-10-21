"""Read-only authorized project/queue/disk and one official HEAD request."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/2b_memory_value_prepare_20261021_01'
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from cluster_io import command


def main():
    url=json.loads((OUT/'media_manifest1920.json').read_text(encoding='utf-8'))[0]['url']
    code="""import getpass,json,pathlib,subprocess,shutil,urllib.request,time
p=pathlib.Path.cwd();allowed=pathlib.Path('/home/u2025010462/share/czy/swt/my_pape').resolve()
assert p.resolve().is_relative_to(allowed) and getpass.getuser()=='u2025010462'
q=subprocess.run(['squeue','-u',getpass.getuser(),'-h','-o','%i %T %j'],capture_output=True,text=True,check=True,timeout=15)
disk=shutil.disk_usage(p);start=time.monotonic();network={}
try:
 with urllib.request.urlopen(urllib.request.Request(OFFICIAL_URL,method='HEAD'),timeout=15) as r:
  network=dict(status=r.status,content_length=r.headers.get('Content-Length'),video_bytes_downloaded=0)
except Exception as e:network=dict(error_type=type(e).__name__,video_bytes_downloaded=0)
print(json.dumps(dict(project=str(p),resolved=str(p.resolve()),user=getpass.getuser(),queue=q.stdout,
 disk_free_bytes=disk.free,official_HEAD=network,HEAD_seconds=time.monotonic()-start,GPU_jobs_submitted=0)))
""".replace('OFFICIAL_URL',repr(url))
    result=command(code,timeout=45)
    with (OUT/'cluster_readonly_check.json').open('x',encoding='utf-8') as f:f.write(result)
    print(result)


if __name__=='__main__':main()
