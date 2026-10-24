"""Check cancellation,32-case plan, exact short-prefix frames and no model import."""
import os,sys,time,importlib.util
from common import ROOT,OUT,HERE,read,save
from source import BoundedAVSource

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
groups=read(OUT/'inputs/groups.json');cases=read(OUT/'inputs/cases.json')
assert len(groups)==8 and len(cases)==32
for group in groups:
    own=[c for c in cases if c['group']==group['id']]
    assert {(c['repeat'],c['config']) for c in own}=={(r,a) for r in range(2) for a in ('slice1','auto4')}
group=next(g for g in groups if g['cohort']=='T98')
checks=[]
for config in ('slice1','auto4'):
    source=BoundedAVSource(ROOT/group['local_media'],group,config=config)
    frames=iter(source.frames());frame=next(frames)
    assert len(source.buffer)==1
    packet=source.take_pair([frame.frame_id])[0];assert packet.frame_id==group['frames'][0]['frame_id']
    next(frames);frames.close();assert not source.buffer and not source.completed
    try:next(source.frames())
    except RuntimeError:pass
    else:raise AssertionError('source restarted after cancellation')
    checks.append(dict(config=config,first_frame_verified=True,cancel_releases=True,single_use=True))
assert 'torch' not in sys.modules
save(OUT/'CPU_preflight.json',dict(status='passed',configs=checks,cases=32,sources=8,
    decord_available=importlib.util.find_spec('decord') is not None,no_model_import=True,GPU_jobs=0))
print(dict(status='passed',configs=2,cases=32),flush=True)
