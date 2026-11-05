"""Minimal middle-layer local-merge adaptation; original encoder untouched."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
for name in ['2b-content-midprune-v1','2b-protection-prototypes-v1']:
    sys.path.insert(0,str(R/'experiments'/name))
from content_prune import encode as original_encode
from anchor_merge import merge_groups

def encode(visual,pixels,grid,*,scorer,enabled=True):
    if not enabled:
        return original_encode(visual,pixels,grid,mode='mid_content',scorer=scorer,return_trace=True)
    pending={};calls=[]
    class Scorer:
        def choose(self,x,shape):
            ids=scorer.choose(x,shape)
            pending['features'],pending['audit']=merge_groups(x,ids,shape)
            return ids
        def resident_bytes(self):return scorer.resident_bytes()
    def inject(module,args,kwargs):
        if len(args)!=1 or 'features' not in pending:raise RuntimeError('unexpected encoder cut interface')
        value=pending.pop('features')
        assert value.shape==args[0].shape
        calls.append(True)
        return (value,),kwargs
    hook=visual.blocks[len(visual.blocks)//2].register_forward_pre_hook(inject,with_kwargs=True)
    try:
        features,ids,trace=original_encode(visual,pixels,grid,mode='mid_content',scorer=Scorer(),return_trace=True)
        assert len(calls)==1 and 'features' not in pending
        trace.update(merge_mass=pending['audit']['mass'].cpu().tolist(),
            source_assignment=pending['audit']['assignment'].cpu().tolist(),local_merge=True)
        return features,ids,trace
    finally:hook.remove();pending.clear()
