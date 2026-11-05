import os,py_compile
from followup import *
from worker import verify
assert os.name!='nt' and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
verify();assert read(OUT/'CPU_local.json')==read(OUT/'CPU_cluster.json')
for p in Path(__file__).parent.glob('*.py'):py_compile.compile(str(p),doraise=True)
media={q['media_path']:q['media_sha256'] for q in read(OUT/'manifest.json')['questions']}
for n,h in media.items():assert sha(R/n)==h,n
save(OUT/'media_verified.json',dict(passed=True,files=len(media),sha_verified=True,
    inherited_causal_sampling_sha256=sha(P/'media_verified.json'),new_full_pixel_replay=False))
print('ready:',len(media),'media;',len(read(OUT/'execution_lock.json')),'locked files; no GPU submitted')
