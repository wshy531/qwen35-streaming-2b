"""CPU scheduler/control-flow simulation; numerical CPU gates run separately."""
from pathlib import Path
import sys,os,types,json
root=Path.cwd();sys.path.insert(0,str(root/'experiments/7b-prune-state-repair-v3'))
import run_prune_tail as m
import torch
m.verify=lambda:None;m.verify_recovery=lambda:None
read=m.read
m.path=lambda q:q;m.sha=lambda q:q['media_sha256'];m.digest=lambda ns:'unchanged'
m.check_pair=lambda *a:None
traces=read(m.PREP/'traces.json')
questions=m.specification('prune')['questions']
m.check_control=lambda r,refs,mode:None
m.torch.cuda.synchronize=lambda:None;m.torch.cuda.empty_cache=lambda:None
arange=torch.arange
def cpu_arange(*a,**kw):
 if kw.get('device')=='cuda':kw['device']='cpu'
 return arange(*a,**kw)
m.torch.arange=cpu_arange
features=torch.zeros(140,4)
model=types.SimpleNamespace(named_parameters=lambda:[],model=types.SimpleNamespace(visual=None),get_video_features=lambda **kw:types.SimpleNamespace(pooler_output=[features]))
m.load_model=lambda:(model,None)
m.readout.pair_inputs=lambda *a,**kw:{'pixel_values_videos':None,'video_grid_thw':None}
m.encode=lambda *a,**kw:(features,arange(140),{'prefix':None})
m.option_ids=lambda p,n:list(range(n));m.forward=lambda *a,**kw:torch.zeros(5)
class Memory:
 def __init__(self,*a,**kw):self.scorer=None
 def append(self,*a,**kw):pass
 def select(self,route):return [types.SimpleNamespace(features=features,metadata=lambda:{})]
class Driver:
 def __init__(self,media,commit):self.commit=commit;self.done=False;self.media=media
 def advance(self,t):
  if not self.done:
   packet=types.SimpleNamespace(rgb=types.SimpleNamespace(shape=(280,392,3)),timestamp=0.);self.commit(packet,packet);self.done=True
  qs=[q for q in questions if q['suite']==self.media['suite'] and q['source_id']==self.media['source_id'] and q['query_time_s']==t]
  assert qs
  ref=traces[self.media['suite']+'|'+self.media['source_id']].get('states',{}).get(qs[0]['id'])
  if ref is None:return {}
  actual=dict(ref);actual['maximum_pending_RGB_bytes']=280*392*3
  return actual
 def close(self):pass
m.Memory=Memory;m.CausalPairs=Driver
os.environ['SLURM_JOB_ID']='CPU_SIMULATION_NOT_A_JOB'
out=root/'runs/7b_prune_state_repair_20261111_01/control_flow_final'
m.main(out)
r=read(out/'completed.json');assert r['answers']==1812 and r['audits']==0 and r['state_checks']==150 and r['control_checks']==12
(out/'SYNTHETIC_ONLY.json').write_text(json.dumps({'GPU':0,'capability_evidence':False,'purpose':'full source/query loop control flow; covers all source and query branches with real frozen states and migrated canvas bytes','classification_rows':1812}))
print('CPU control-flow simulation passed: 1812 rows, no GPU or real answers')
