import subprocess
from fair_common import *
verify();assert read(P/'CPU4_local.json')==read(P/'CPU4_cluster.json')
m=read(P/'media_verified.json');assert m['passed'] and m['sources']==144 and m['questions']==150 and m['fresh_pixel_queries']==2
subprocess.run(['bash','-n',str(Path(__file__).parent/'run_quality_v3.sh')],check=True)
save(P/'cluster_ready4.json',dict(passed=True,CPU_equal=True,media_files=144,source_states=150,fresh_pixel_queries=2,vision_pairs=m['vision_pairs'],lock_sha256=sha(P/'execution_lock4.json')))
print('READY4',m['vision_pairs'],sha(P/'execution_lock4.json'))
