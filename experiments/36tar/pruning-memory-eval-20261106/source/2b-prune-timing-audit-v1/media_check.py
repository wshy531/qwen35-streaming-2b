"""CPU-only exact old pixel/PTS replay; no model load or GPU."""
from collections import deque
from audit_common import *
def main():
    assert sys.platform!='win32'
    data=ROOT/'datasets/StreamingBench-eval200-v1/videos';total=0;rows=[]
    media={x['name']:x for x in read(PREP/'media.json')}
    contracts={x['id']:x for x in read(PREP/'sample_contract.json')}
    for q in read(PREP/'requests.json'):
        p=data/q['media_name'];m=media[p.name]
        assert p.resolve().parent==data.resolve() and p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
        recent=deque(maxlen=16);count=0
        def receive(a,b,padding):
            nonlocal count
            for i,f in enumerate((a,b)):
                recent.append(dict(pair_index=count,within_pair=i,frame_id=f.frame_id,pts_s=f.timestamp,
                    rgb_sha256=hashlib.sha256(f.rgb.tobytes()).hexdigest(),padding=padding))
            count+=1
        driver=V0Pairs(p,receive,config='auto4')
        try:state=driver.advance(q['query_time_s'])
        finally:driver.close()
        c=contracts[q['id']];assert state==c['state'] and count==c['pairs'] and list(recent)==c['frames']
        total+=count;rows.append(dict(id=q['id'],pairs=count,verified_recent_frames=len(recent)))
        print(q['media_name'],count,flush=True)
    assert total==513
    save(PREP/'CPU_media_cluster.json',dict(status='passed',sources=6,replayed_pairs=total,rows=rows,GPU=0))
if __name__=='__main__':main()
