"""CPU-only sensitivity analysis of saved gradients, no model updates."""
import argparse
import json
from pathlib import Path
import torch


def first_adam_step(g):
    # Analytic first step of former CE recipe: fresh AdamW, lr=1e-4,
    # weight_decay=0, eps=1e-8, global clipping at 1. No momentum history.
    g=g*min(1.,1./(float(g.norm())+1e-6))
    return -1e-4*g/(g.abs()+1e-8)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    rows=[]
    for path in sorted(args.folder.glob('*-gradients.pt')):
        tensors=torch.load(path,map_location='cpu',weights_only=True)
        vectors={k:torch.cat([v[n].flatten().double() for n in sorted(v)]) for k,v in tensors.items()}
        base=vectors['whole_first'];details={}
        for name in ('whole_repeat','recompute'):
            value=vectors[name];delta=value-base
            step_delta=first_adam_step(value)-first_adam_step(base)
            cosine=float(torch.nn.functional.cosine_similarity(base,value,dim=0,eps=1e-30))
            details[name]=dict(gradient_norm=float(value.norm()),difference_l2=float(delta.norm()),
                relative_l2=float(delta.norm()/base.norm()),cosine=cosine,
                sign_changes=int(((value>0)!=(base>0)).sum()),
                analytic_fresh_adam_step_max_abs_difference=float(step_delta.abs().max()),
                analytic_fresh_adam_step_rms_difference=float(step_delta.square().mean().sqrt()))
        violations=[]
        for name,ref in tensors['whole_first'].items():
            val=tensors['recompute'][name];repeat=tensors['whole_repeat'][name]
            mask=(val-ref).abs()>2e-6+5e-4*ref.abs()
            for index in mask.nonzero():
                at=tuple(index.tolist())
                violations.append(dict(parameter=name,index=list(at),whole_first=float(ref[at]),
                    whole_repeat=float(repeat[at]),recompute=float(val[at]),
                    allowed_abs_error=float(2e-6+5e-4*ref[at].abs())))
        rows.append(dict(file=path.name,parameters=base.numel(),whole_first_norm=float(base.norm()),
                         comparisons=details,original_gate_violations=violations))
    result=dict(cases=rows,original_gate_not_changed=True,model_updates=0,GPU=0,
        limitation='Post hoc sensitivity only. Analytic fresh-state AdamW is not a real update or a multi-step training guarantee.')
    with (args.folder/'gradient_sensitivity.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
