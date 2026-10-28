"""Fixed CPU old-layout regression and high-resolution bounded-state checks."""
from dataclasses import replace
import os
import sys
import torch
from sember_common import ROOT,OUT,save,sha
from sember_nodes import pair_node,merge,validate
from sember_banks import Bank,Banks,select


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(20261025)
    sys.path.extend(str(ROOT/'experiments'/p) for p in ('2b-compressed-banks-v1','2b-natural-eval-v1',
        '2b-long-stream-v1','2b-ls-depth-v1','2b-closed-balance-v1','2b-short-event-v1'))
    from policy_adapters import BoundedForest,BoundedShared
    from features import raw_block
    old=[];new=[]
    for policy in ('M2','A06','LS'):
        for size in (48,24):
            old.append(BoundedShared(size) if policy=='LS' else BoundedForest(policy=='A06',size))
            new.append(Bank(policy,size,budget=384,recent=192,closed=144))
    checked=0
    for i in range(96):
        x=torch.randn(48,32);starts=i%5==0 or i%17==0
        for a,b in zip(old,new):
            ids=torch.arange(48) if b.pair_tokens==48 else torch.arange(1,48,2)
            block=raw_block(x,i,i*4.,i*4.+2)
            block=replace(block,features=block.features[ids].clone(),mass=block.mass[ids].clone(),xy=block.xy[ids].clone())
            a.append(block,starts_event=starts)
            b.append(pair_node(x,i,i*4.,i*4.+2,(6,8),ids),starts_event=starts)
            aa,bb=a.export(),b.export();assert len(aa)==len(bb)
            for left,right in zip(aa,bb):
                for field in ('first','last','begin','end','center','groups','depth'):assert getattr(left,field)==getattr(right,field),field
                for field in ('features','mass','coords'):assert torch.equal(getattr(left,field),getattr(right,field)),field
            for field in ('tokens','recent_tokens','open_tokens','closed_tokens','history_tokens','within_merges','long_merges'):
                assert a.status().get(field,0)==b.status().get(field,0),(i,b.policy,field)
            checked+=1
    banks=Banks();peaks={a:0 for a in banks.banks};max_depth=0
    for i in range(160):
        x=torch.randn(108,32) if i%7==0 else x+.02*torch.randn(108,32)
        trace=banks.append(x,i,i*4.,i*4.+2)
        for arm,bank in banks.banks.items():
            state=bank.status();peaks[arm]=max(peaks[arm],state['tokens'])
            assert state['tokens']<=bank.budget
            for node in bank.nodes():validate(node);max_depth=max(max_depth,node.depth)
        assert trace['pair_index']==i
    assert peaks['M2_change54']<=810 and peaks['A06_change54']<=810
    assert peaks['LS_full108']==peaks['LS_change54']==864
    before=banks.fingerprint();copied=banks.export('LS_full108');copied[0].features.zero_()
    assert banks.fingerprint()==before
    for mode,ids in select(torch.zeros(108,32),torch.zeros(108,32))[0].items():
        assert len(ids)==(108 if mode=='full' else 54) and bool(torch.all(ids[1:]>ids[:-1]))
    # Original-grid sorting must stay correct when source rows exceed 6 and columns exceed 8.
    left=pair_node(torch.randn(108,32),0,0.,2.,(9,12));right=pair_node(torch.randn(108,32),1,4.,6.,(9,12))
    joined=merge(left,right);validate(joined);assert float(joined.mass.sum())==216
    wide=Banks();wide.append(torch.randn(108,2048),0,0.,2.)
    assert wide.status()['shared_previous_bytes']==884736
    banks.clear();wide.clear();assert not banks.banks and banks.previous is None
    result=dict(status='passed',old_48_layout_step_comparisons=checked,high_resolution_pair_steps=160,
        peaks_visual_tokens=peaks,max_merge_depth_exercised=max_depth,
        export_isolation=True,zero_vector_stable_selection=True,dynamic_grid_sort_and_mass=True,
        previous_full_pair_bytes_FP32_2048=884736,pretrained_loads=0,GPU_jobs=0,
        no_capability_or_speed_claim=True,
        old_source_SHA={p:sha(ROOT/p) for p in ('experiments/2b-compressed-banks-v1/policy_adapters.py',
            'experiments/2b-natural-eval-v1/forest.py','experiments/2b-ls-depth-v1/shared.py')})
    OUT.mkdir(exist_ok=True);save(OUT/'core_preflight.json',result);print(result,flush=True)


if __name__=='__main__':main()
