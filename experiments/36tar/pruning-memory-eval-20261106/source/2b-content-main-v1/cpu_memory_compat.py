"""Preparation only: 54-group nodes into existing M0/M2/A06/LS, no new QA/GPU."""
import os,json,argparse,sys
from pathlib import Path
from common import ROOT,PREP,runtime,save
def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261101)
    from sember_nodes import Node,validate,digest
    from sember_banks import Bank
    modes={}
    for policy in ('M0','M2','A06','LS'):
        for protocol in ('same_coverage_scale','same_token_cap'):
            factor=.5 if protocol=='same_coverage_scale' else 1.
            modes[policy+'/'+protocol]=Bank(policy,54,budget=int(864*factor),recent=int(432*factor),closed=int(324*factor))
    rows=[]
    with t.inference_mode():
        for i in range(80):
            # Deterministic varying sparse positions, complete original 2x2 group IDs.
            ids=(t.arange(54)*2+(i%2)).sort().values
            coords=t.stack((t.full_like(ids,i),ids//12,ids%12),1)
            node=Node(t.randn(54,32),t.ones(54,1),coords,i,i,float(i*4),float(i*4+2),float(i*4+1),grid=(9,12))
            validate(node)
            # Synthetic fixed event flags test the interface only; not event recognition.
            for name,bank in modes.items():
                bank.append(node,starts_event=i%5==0)
                status=bank.status();assert status['tokens']<=bank.budget
                assert all(n.end<=i*4+2 for n in bank.nodes())
                for n in bank.nodes():validate(n)
                before=digest(bank.nodes());copies=bank.export()
                copies[-1].features.zero_();assert digest(bank.nodes())==before
        for name,bank in modes.items():
            rows.append(dict(name=name,**bank.status()))
    save(out,dict(status='passed',synthetic_pairs=80,policies=4,protocols=2,bank_updates=640,
        CPU_only=True,rows=rows,GPU=0,real_media=0,real_QA=0,capability_claim=False,
        event_flags_synthetic=True,original_coordinates_preserved=True,
        integration_pending_real_event_descriptor_and_coverage_controls=True,
        API33_exact_108_recent_interface_not_reused=True))
    print('54-group M0/M2/A06/LS: 640 synthetic updates, bounded state and copy isolation passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
