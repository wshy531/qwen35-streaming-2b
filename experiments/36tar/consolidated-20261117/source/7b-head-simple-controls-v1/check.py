"""Real tiny interface plus complete synthetic execution; never capability evidence."""
import argparse,importlib.util,types,contextlib,io,os
import run_controls as r
def main(out):
    r.verify_controls();r.torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('tiny',r.ROOT/'experiments/7b-prune-selector-transfer-v1/preflight.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main(out)
    tasks=r.read(r.PREP/'tasks.json');traces=r.read(r.HERE/'traces.json');states={};hashes={};active={}
    for t in tasks:
        q=t['question'];s=dict(traces[q['suite']+'|'+q['source_id']]['states'][q['id']]);s['maximum_pending_RGB_bytes']=280*392*3
        states[str(r.path(q)),q['query_time_s']]=(s,t);hashes[str(r.path(q))]=q['media_sha256']
        for c in t['methods'].values():assert c==sorted(set(c)) and all(0<=i<t['n_history'] for i in c)
        for c in t['todo']:assert not any(e['choice']==c for e in t['cached'])
    realread=r.read;realsha=r.sha
    def read(p):
        if p==r.PREP/'tasks.json':
            tt=r.json.loads(r.json.dumps(tasks))
            for t in tt:
                q=t['question']
                for e in t['cached']+([t['reference']] if t['reference'] else []):
                    e['row'].update(option_logits=list(map(float,range(len(q['options'])))),
                        selected_original=(len(q['options'])-1+e['row']['shift'])%len(q['options']),visual_tokens=70*(4+len(e['choice'])))
            return tt
        return realread(p)
    class Mem:
        def __init__(self,*a,**k):self.archive=None;self.scorer=None;self.bank=types.SimpleNamespace(nodes=lambda:[])
        def append(self,*a):pass
        def parts(self):
            node=lambda:types.SimpleNamespace(features=r.torch.zeros(70,4))
            return [node() for _ in range(4)],[node() for _ in range(active['n_history'])]
    class Driver:
        def __init__(self,media,commit):self.media=str(media);self.commit=commit;self.started=False
        def advance(self,t):
            if not self.started:
                import numpy as np
                a=types.SimpleNamespace(rgb=np.zeros(r.NEW_CANVAS,dtype=np.uint8),timestamp=0.)
                self.commit(a,a);self.started=True
            state,task=states[self.media,t];active.update(task);return state
        def close(self):pass
    r.read=read;r.sha=lambda p:hashes.get(str(p)) or realsha(p)
    r.Memory=Mem;r.CausalPairs=Driver;r.check_pair=lambda *a:None;r.digest=lambda x:'synthetic'
    r.readout=types.SimpleNamespace(pair_inputs=lambda *a,**k:dict(pixel_values_videos=None,video_grid_thw=None))
    r.encode=lambda *a,**k:(r.torch.zeros(70,4),r.torch.arange(70),{})
    model=r.torch.nn.Linear(4,4).eval().requires_grad_(False);model.model=types.SimpleNamespace(visual=None)
    r.load_model=lambda:(model,None);r.forward=lambda *a,**k:r.torch.arange(5).float();r.option_ids=lambda p,n:list(range(n))
    r.torch.cuda.synchronize=lambda:None;r.gc.collect=lambda:0
    os.environ['SLURM_JOB_ID']='SYNTHETIC_ONLY';flow=out.parent/(out.stem+'_FLOW_SYNTHETIC_ONLY')
    with contextlib.redirect_stdout(io.StringIO()):r.main(flow)
    done=realread(flow/'completed.json');rr=[r.json.loads(s) for s in (flow/'answers.jsonl').read_text().splitlines()]
    assert len(rr)==len({(x['id'],x['route'],x['shift']) for x in rr})==2520
    assert done['classifications']==294 and done['references']==12
    v=realread(out);v.update(full_flow=done,rows=2520,synthetic_only=True,lock_sha=realsha(r.PREP/'lock.json'));r.save(out,v)
    print(r.json.dumps(dict(passed=True,rows=2520,actual_new_classifications=294,real_capability_answers=0)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=r.Path,required=True);main(a.parse_args().out)
