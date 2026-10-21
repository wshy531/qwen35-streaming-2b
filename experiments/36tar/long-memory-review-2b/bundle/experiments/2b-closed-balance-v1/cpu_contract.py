"""Fixed six real boundary schedules, synthetic32-wide values, no capability claim."""
import hashlib
import json
from pathlib import Path
import sys
import time

from policy import ROOT,DepthForest384
from audit_nodes import OUT
from memory384 import Forest384
from features import raw_block
import torch

sys.path.insert(0,str(ROOT/'experiments/2b-long-stream-v1'))
from online_bank import nodes_digest


def digest(bank):return nodes_digest(bank.export())


def main():
    start=time.monotonic();torch.set_num_threads(1);generator=torch.Generator().manual_seed(123)
    logs=ROOT/'runs/2b_long_stream_prepare_20261020_01/gpu_217776/memory/events.jsonl'
    streams={};queries={}
    for line in logs.open(encoding='utf-8'):
        e=json.loads(line)
        if e['arm']!='M2':continue
        if e['event']['kind']=='pair':streams.setdefault(e['source_path'],[]).append(dict(**e['event'],starts_event=e['state']['bank']['starts_event']))
        elif e['event']['kind']=='query':queries.setdefault(e['source_path'],{}).setdefault(e['event']['committed_pairs'],[]).append(e['event']['query_id'])
    results=[];layout=[];max_bytes=0
    for source,frames in streams.items():
        original,compat,candidate=Forest384(),DepthForest384(original_selector=True),DepthForest384()
        for e in frames:
            i=e['pair_index'];x=torch.randn(48,32,generator=generator)
            block=raw_block(x,i,e['begin_s'],e['end_s'])
            for bank in (original,compat,candidate):bank.append(block,starts_event=e['starts_event'])
            assert digest(original)==digest(compat),'original-selector clone drifted'
            old,new=original.status(),candidate.status()
            assert old['tokens']==new['tokens']<=384
            assert old['within_merges']==new['within_merges'] and old['long_merges']==new['long_merges']
            assert old['recent_event_starts']==new['recent_event_starts']
            assert all(torch.equal(a.features,b.features) and torch.equal(a.coords,b.coords)
                for (a,_),(b,_) in zip(original.recent,candidate.recent))
            if original.open is not None:assert nodes_digest([original.open])==nodes_digest([candidate.open])
            nodes=candidate.nodes();assert nodes[0].first==0 and nodes[-1].last==i
            assert all(a.last+1==b.first for a,b in zip(nodes,nodes[1:]))
            assert abs(sum(float(n.mass.sum()) for n in nodes)-48*(i+1))<1e-3
            assert len(candidate.recent)<=4 and len(candidate.roots)<=3
            assert set(candidate.__dict__)==set(compat.__dict__), 'unexpected candidate archive'
            max_bytes=max(max_bytes,new['resident_tensor_bytes'])
            assert new['resident_tensor_bytes']<=384*(32*4+4+3*8)
            for ident in queries[source].get(i+1,[]):
                layout.append(dict(id=ident,source=source,policy='A06',nodes=[n.metadata() for n in candidate.export()]))
        before=digest(candidate);copy=candidate.export();copy[0].features.fill_(123)
        assert digest(candidate)==before,'export aliasing'
        for block,flag in [(raw_block(torch.zeros(48,32),len(frames)-1,10000,10001),True),
                           (raw_block(torch.zeros(48,32),len(frames),10000,10001),'yes')]:
            try:candidate.append(block,starts_event=flag)
            except ValueError:pass
            else:raise AssertionError('invalid input accepted')
            assert digest(candidate)==before
        results.append(dict(source=source,pairs=len(frames),
            original_synthetic_depths=[n.depth for n in original.nodes()],
            candidate_synthetic_depths=[n.depth for n in candidate.nodes()],
            candidate_closed_selections=candidate.selection_count,
            original_compat_bitwise_equal=True,payloads_released_per_source=True))
        del original,compat,candidate,copy,block,x
    assert sum(r['pairs'] for r in results)==1426 and len(layout)==34
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',pairs=1426,sources=6,query_layouts=layout,streams=results,
        event_flags_are_actual_causal_recorded_flags=True,features_are_random32_NOT_real_model=True,
        max_resident_tensor_bytes_32wide=max_bytes,max_resident_tensor_bytes_2048wide=384*(2048*4+4+3*8),
        descriptor_if_live_detector_added_bytes=8192,selector_persistent_fields='two scalar flags/counters only; no extra tensors',
        all_recent_open_bitwise_same=True,range_and_mass_conserved=True,copy_and_rejection_contracts_pass=True,
        pretrained_loads=0,GPU_jobs=0,seconds=time.monotonic()-start,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        stop='one fixed CPU first version; no metric/order sweeps',
        limits='depths here come from synthetic features; no real2B capability or compression-quality improvement established')
    with (OUT/'CPU_contract1426.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(dict(status='passed',pairs=1426,query_layouts=34,seconds=time.monotonic()-start,
        max_feature_bytes2048=result['max_resident_tensor_bytes_2048wide']))


if __name__=='__main__':main()
