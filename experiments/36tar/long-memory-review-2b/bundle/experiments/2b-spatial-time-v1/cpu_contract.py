"""All1426 recorded arrivals, five fixed policies, random32 features only."""
import hashlib
import json
from pathlib import Path
import sys
import time

from policy import ROOT,FixedForest,RecentBank,raw_block
from memory384 import Forest384
from review_media import OUT,save
import torch

sys.path.insert(0,str(ROOT/'experiments/2b-long-stream-v1'))
from online_bank import nodes_digest

ARMS=('M0_384','M2_384','M0_768','M2_768','S24_384')


def main():
    start=time.monotonic();torch.set_num_threads(1);gen=torch.Generator().manual_seed(123)
    log=ROOT/'runs/2b_long_stream_prepare_20261020_01/gpu_217776/memory/events.jsonl';streams={};queries={}
    for line in log.open(encoding='utf-8'):
        e=json.loads(line)
        if e['arm']!='M2':continue
        if e['event']['kind']=='pair':streams.setdefault(e['source_path'],[]).append(dict(**e['event'],flag=e['state']['bank']['starts_event']))
        elif e['event']['kind']=='query':queries.setdefault(e['source_path'],{}).setdefault(e['event']['committed_pairs'],[]).append(e['event']['query_id'])
    peak_bytes={a:0 for a in ARMS};fixtures=[];counts=0
    for source,events in streams.items():
        banks=dict(M0_384=RecentBank(384),M2_384=FixedForest(budget=384),M0_768=RecentBank(768),
            M2_768=FixedForest(budget=768),S24_384=FixedForest(budget=384,old_tokens=24));original=Forest384()
        for e in events:
            i=e['pair_index'];block=raw_block(torch.randn(48,32,generator=gen),i,e['begin_s'],e['end_s'])
            original.append(block,starts_event=e['flag'])
            for arm,bank in banks.items():
                bank.append(block) if arm.startswith('M0') else bank.append(block,starts_event=e['flag'])
                state=bank.status();assert state['tokens']<=bank.budget
                peak_bytes[arm]=max(peak_bytes[arm],state['resident_tensor_bytes'])
                assert state['resident_tensor_bytes']<=bank.budget*(32*4+4+3*8)
            assert nodes_digest(original.export())==nodes_digest(banks['M2_384'].export())
            recent=[n for n,_ in original.recent]
            for arm in ('M0_384','M2_384','M0_768','M2_768','S24_384'):
                ns=banks[arm].nodes();tail=ns[-len(recent):]
                assert nodes_digest(tail)==nodes_digest(recent),'recent full features changed'
                if not arm.startswith('M0'):
                    assert ns[0].first==0 and ns[-1].last==i
                    assert all(a.last+1==b.first for a,b in zip(ns,ns[1:]))
                    assert abs(sum(float(n.mass.sum()) for n in ns)-48*(i+1))<1e-3
            s24=banks['S24_384'];assert s24.spatial_reductions==max(0,i+1-4)
            assert all(len(n.features)==24 for n in s24.roots+([] if s24.open is None else [s24.open]))
            for ident in queries[source].get(i+1,[]):
                fixtures.append(dict(id=ident,source=source,arms={arm:[dict(metadata=n.metadata(),
                    coords=n.coords.tolist(),mass=n.mass.flatten().tolist()) for n in bank.nodes()] for arm,bank in banks.items()}))
            counts+=1
        for bank in banks.values():
            before=nodes_digest(bank.export());copy=bank.export();copy[0].features.fill_(99)
            assert nodes_digest(bank.export())==before
            bad=raw_block(torch.zeros(48,32),len(events)-1,10000,10001)
            try:bank.append(bad) if isinstance(bank,RecentBank) else bank.append(bad,starts_event=True)
            except ValueError:pass
            else:raise AssertionError('duplicate accepted')
            assert nodes_digest(bank.export())==before
        del banks,original,copy,block
    assert counts==1426 and len(fixtures)==34 and not torch.cuda.is_initialized()
    save(OUT/'CPU_policy1426.json',dict(status='passed',pairs=counts,sources=6,fixtures=fixtures,
        original384_bitwise_compatible=True,all_recent_full_bitwise_same=True,range_mass_copy_rejection_passed=True,
        max_bytes32=peak_bytes,max_bytes2048={a:(768 if a.endswith('768') else 384)*(2048*4+4+3*8) for a in ARMS},
        shared_descriptor_live_bytes=8192,pretrained_loads=0,GPU_jobs=0,seconds=time.monotonic()-start,
        features='synthetic32; NOT real2B compression/readability evidence',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    print(dict(status='passed',pairs=counts,query_fixtures=34,seconds=time.monotonic()-start))


if __name__=='__main__':main()
