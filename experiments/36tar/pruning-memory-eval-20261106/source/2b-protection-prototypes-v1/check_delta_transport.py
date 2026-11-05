"""State Soup-style transport adapted to ONE frozen-input GDN recurrence.

No model loading/training; does NOT establish multilayer hybrid equivalence.
"""
import os,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-sember-v1'))
from sember_common import runtime
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
from qwen35_local.modeling_qwen3_5 import torch_recurrent_gated_delta_rule as recurrent

def transport(delta,key,g,beta):
    """A_R (S_history-S_prefix), with recent-input keys/gates held fixed."""
    d=delta.clone()
    for i in range(key.shape[1]):
        k=key[:,i];a=g[:,i].exp()[...,None,None];b=beta[:,i][...,None]
        d=d*a;read=(d*k[...,None]).sum(-2)
        d=d-k[...,None]*(b*read)[...,None,:]
    return d

records=[]
for steps in (8,32,128,432):
    q=t.randn(1,steps,2,16);k=t.nn.functional.normalize(t.randn_like(q),dim=-1)
    v=t.randn(1,steps,2,16);g=-.02*t.rand(1,steps,2);beta=t.sigmoid(t.randn(1,steps,2))
    prefix=t.randn(1,2,16,16);history=t.randn_like(prefix)
    _,recent=recurrent(q,k,v,g,beta,prefix,True)
    _,target=recurrent(q,k,v,g,beta,history,True)
    corrected=recent+transport(history-prefix,k,g,beta)
    naive=.5*(recent+history)
    error=float((corrected-target).abs().max())
    assert t.allclose(corrected,target,atol=2e-6,rtol=2e-5),(steps,error)
    records.append(dict(steps=steps,transport_max_abs_error=error,
        transport_relative_L2=float((corrected-target).norm()/target.norm()),
        recent_only_relative_L2=float((recent-target).norm()/target.norm()),
        naive_mean_relative_L2=float((naive-target).norm()/target.norm()),
        surviving_history_delta_norm_fraction=float(transport(history-prefix,k,g,beta).norm()/(history-prefix).norm())))
out=R/'runs/delta_transport_mechanism_20261103_01';out.mkdir(exist_ok=False)
(out/'CPU.json').write_text(json.dumps(dict(passed=True,rows=records,GPU=0,pretrained_loads=0,
    source='State Soup section 2.3, adapted to fixed-input GDN recurrence',
    limitations=['Keys/gates/values fixed, no convolution boundary reconstruction',
        'Layer inputs in a complete hybrid network depend on earlier layers; not equivalent to joint prefill',
        'No video quality evidence, no state mixing added to default or running experiment']),indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
