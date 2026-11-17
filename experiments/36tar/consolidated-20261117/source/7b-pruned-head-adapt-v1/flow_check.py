"""SYNTHETIC_ONLY orchestration test; real CPU numerical checks live in preflight."""
import argparse,contextlib,io,types,os
import run_adapt as r

def main(out):
    r.torch.set_num_threads(1);r.verify_adapt()
    realread=r.read;realsha=r.sha;groups=realread(r.HERE/'fit_sources.json')
    tasks=realread(r.HERE/'questions.json');traces=realread(r.HERE/'traces.json')
    states={};media_hash={};frozen=False
    for g in groups:
        for q in g['questions']:media_hash[str(r.path(q))]=q['media_sha256']
    for q in tasks:
        media_hash[str(r.path(q))]=q['media_sha256']
        s=dict(traces[q['suite']+'|'+q['source_id']]['states'][q['id']]);s['maximum_pending_RGB_bytes']=280*392*3
        states[str(r.path(q)),q['query_time_s']]=s
    expected=[]
    for q in tasks:
        for route in ('recent8','LS','head'):
            for shift in (0,2):
                expected.append(dict(id=q['id'],route=route,shift=shift,option_logits=[float(i) for i in range(len(q['options']))],
                    selected_original=(len(q['options'])-1+shift)%len(q['options']),visual_tokens=280))
    def read(p):
        if p==r.HERE/'questions.json':assert frozen,'evaluation opened before freeze'
        if p==r.HERE/'references.json':return expected
        if p==r.HERE/'old_choices.json':return {q['id']:[] for q in tasks}
        return realread(p)
    class Memory:
        def __init__(self,*a,**k):
            self.archive=None;self.scorer=None;self.bank=types.SimpleNamespace(nodes=lambda:[])
            self.nodes=[types.SimpleNamespace(features=r.torch.zeros(70,4)) for _ in range(4)]
        def append(self,*a):pass
        def parts(self):return self.nodes,[]
    class Driver:
        def __init__(self,media,commit):self.media=str(media);self.commit=commit;self.started=False
        def advance(self,t):
            if not self.started:
                import numpy as np
                a=types.SimpleNamespace(rgb=np.zeros(r.NEW_CANVAS,dtype=np.uint8),timestamp=0.)
                self.commit(a,a);self.started=True
            return states.get((self.media,t),{})
        def close(self):pass
    def train(fs,ss,gs,dest):
        nonlocal frozen
        assert len(fs)==176 and {f['id'] for f in fs}=={q['id'] for g in groups for q in g['questions']}
        assert not {f['id'] for f in fs}&{q['id'] for q in tasks}
        (dest/'head-visible.pt').write_bytes(b'SYNTHETIC_ONLY');frozen=True;return None
    r.read=read;r.sha=lambda p:media_hash.get(str(p),None) or realsha(p)
    r.Memory=Memory;r.CausalPairs=Driver;r.check_pair=lambda *a:None;r.digest=lambda x:'synthetic'
    r.readout=types.SimpleNamespace(pair_inputs=lambda *a,**k:dict(pixel_values_videos=None,video_grid_thw=None))
    r.encode=lambda *a,**k:(r.torch.zeros(70,4),r.torch.arange(70),{})
    model=r.torch.nn.Linear(4,4).eval().requires_grad_(False);model.model=types.SimpleNamespace(visual=None)
    r.load_model=lambda:(model,None);r.load_head=lambda:None
    r.forward=lambda *a,representation=False,**k:r.torch.zeros(4) if representation else r.torch.arange(5).float()
    r.option_ids=lambda p,n:list(range(n))
    r.descriptor_features=lambda *a:dict(query=r.torch.zeros(4),recent=r.torch.zeros(8),history=r.torch.zeros(4,8),metadata=r.torch.zeros(4,8),node_mask=r.torch.zeros(4,dtype=r.torch.bool))
    r.train_head=train;r.select_nodes=lambda h,q,recent,history:(recent,[],{})
    r.torch.cuda.synchronize=lambda:None;r.gc.collect=lambda:0
    os.environ['SLURM_JOB_ID']='SYNTHETIC_ONLY'
    with contextlib.redirect_stdout(io.StringIO()):r.main(out)
    value=realread(out/'completed.json');assert value['eval_questions']==210 and value['references']==1260
    r.save(out/'FLOW_SYNTHETIC_ONLY.json',dict(passed=True,real_model_answers=0,**value))
    print(r.json.dumps(dict(passed=True,SYNTHETIC_ONLY=True,train=176,evaluation=210,references=1260)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=r.Path,required=True);main(a.parse_args().out)
