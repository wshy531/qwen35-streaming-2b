"""Exercise actual 453-pair scheduler and negative controls, without weights."""
from pathlib import Path
from collections import deque,defaultdict
import argparse,copy,json,os
from prepare import ROOT,read,save
from scheduler import schedule
EXEC=ROOT/'runs/read_benefit_execution_prepare_20261028_01'
def main(output):
    q=read(EXEC/'feature_requests.json');r=read(EXEC/'media_resolution.json');s=read(EXEC/'sources.json');plans=schedule(q,r,s)
    seen=set();max_live=0;count=0
    for plan in plans:
        live=deque(maxlen=4);at=defaultdict(list)
        for j in plan['queries']:at[j['last_pair']].append(j)
        for pair in plan['source']['pairs']:
            live.append(pair['index']);count+=1;max_live=max(max_live,len(live))
            for job in at[pair['index']]:
                assert list(live)==job['pair_indices'];seen.add(job['request']['id'])
    assert len(seen)==120 and count==453 and max_live==4
    checks={}
    def reject(name,qq,rr,ss):
        try:schedule(qq,rr,ss)
        except ValueError:checks[name]=True
        else:raise AssertionError(name)
    bad=copy.deepcopy(q);bad[0]['options']=['LEAK'];reject('options_rejected',bad,r,s)
    bad=copy.deepcopy(q);bad[0]['query_time_s']=-1;reject('future_frames_rejected',bad,r,s)
    bad=copy.deepcopy(r);bad[q[0]['id']]['sha256']='0'*64;reject('wrong_source_SHA_rejected',q,bad,s)
    bad=copy.deepcopy(s);bad[0]['pairs'].pop(0);reject('missing_required_pair_rejected',q,r,bad)
    bad=copy.deepcopy(q);bad.append(bad[0]);reject('duplicate_request_rejected',bad,r,s)
    bad=copy.deepcopy(s);bad[0]['pairs']=list(reversed(bad[0]['pairs']));reject('reverse_time_rejected',q,r,bad)
    save(output,dict(status='passed_CPU_feature_schedule',sources=len(plans),queries=len(seen),visual_pairs=count,formal_features=240,
        warmup_features=2,max_resident_visual_pairs=max_live,negative_controls=checks,real_model_loaded=False,GPU_jobs=0))
    print(dict(status='passed_CPU_feature_schedule',queries=len(seen),visual_pairs=count,negative_controls=len(checks)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
