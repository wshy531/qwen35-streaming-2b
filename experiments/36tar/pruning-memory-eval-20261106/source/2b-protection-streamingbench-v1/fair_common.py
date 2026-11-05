from common import *
ARMS=('recent','recent864','joint','isolated')
def verify():
    for n,h in read(P/'execution_lock4.json').items():
        p=R/n
        if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
        assert sha(p)==h,n
