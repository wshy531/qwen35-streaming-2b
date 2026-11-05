import subprocess
from worker import *
verify();assert read(P/'CPU_local.json')==read(P/'CPU_cluster.json')
plan=read(OLD/'protocol.json')
for s in read(OLD/'paired_sources.json'):
    p=Path(plan['paired_media_root'])/s['name'];assert sha(p)==s['media_sha256']
subprocess.run(['bash','-n',str(Path(__file__).parent/'run_diagnosis.sh')],check=True)
save(P/'cluster_ready.json',dict(passed=True,CPU_equal=True,media_files=5,lock_sha256=sha(P/'execution_lock.json')))
print('READY',sha(P/'execution_lock.json'))
