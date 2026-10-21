"""7200 synthetic pairs simulate4h at1FPS, not real video/model capability."""
import hashlib,json,random,time
from pathlib import Path
import psutil,torch
from memory_policies import ROOT,FixedForest,RecentBank,Reservoir768,raw_block
OUT=ROOT/'runs/2b_memory_value_prepare_20261021_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(1);torch.manual_seed(20261021)
    start=time.monotonic();rss=0;banks=dict(recent=RecentBank(768),reservoir=Reservoir768(),M2=FixedForest(budget=768))
    ref_rng=random.Random(20261020);reference=[];promoted=0;checkpoints=[];peak={a:0 for a in banks}
    for i in range(7200):
        block=raw_block(torch.randn(48,32),i,i*2.,i*2.+1.)
        banks['recent'].append(block);banks['reservoir'].append(block)
        banks['M2'].append(block,starts_event=i==0 or i%17==0 or i%113==0)
        if i>=4:
            promoted+=1;old=i-4
            if len(reference)<12:reference.append(old)
            else:
                j=ref_rng.randrange(promoted)
                if j<12:reference[j]=old
        expected=sorted(reference+list(range(max(0,i-3),i+1)))
        assert banks['reservoir'].status()['retained_pairs']==expected
        assert [n.first for n in banks['recent'].nodes()]==list(range(max(0,i-15),i+1))
        for a,b in banks.items():
            nodes=b.nodes();tokens=sum(len(n.features) for n in nodes);size=sum(n.metadata()['tensor_bytes'] for n in nodes)
            assert tokens<=768 and size<=768*(32*4+4+3*8)
            assert [n.first for n in nodes[-min(i+1,4):]]==list(range(max(0,i-3),i+1))
            peak[a]=max(peak[a],size)
        ns=banks['M2'].nodes()
        assert ns[0].first==0 and ns[-1].last==i and all(a.last+1==b.first for a,b in zip(ns,ns[1:]))
        assert abs(sum(float(n.mass.sum()) for n in ns)-48*(i+1))<1e-3
        if i in (239,1425,3599,7199):
            checkpoints.append(dict(pairs=i+1,arms={a:[n.metadata() for n in b.nodes()] for a,b in banks.items()}))
        if i%100==0:
            rss=max(rss,psutil.Process().memory_info().rss)
            assert rss<2**30 and time.monotonic()-start<240
    for a,b in banks.items():
        copied=b.export();before=b.nodes()[0].features.clone();copied[0].features.fill_(123)
        assert torch.equal(before,b.nodes()[0].features)
        bad=raw_block(torch.zeros(48,32),7199,14401.,14402.)
        try:b.append(bad,starts_event=True) if a=='M2' else b.append(bad)
        except ValueError:pass
        else:raise AssertionError('duplicate accepted')
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',pairs=7200,nominal_engineering_hours=4,synthetic_feature_width=32,
        original_visual_tokens_processed=7200*48,fixed_visual_budget=768,
        standalone_algorithmR_reference_matches=True,mass_range_copy_duplicate_checks_passed=True,
        peak_live_bank_tensor_bytes32=peak,max_live_bank_tensor_bytes2048=768*(2048*4+4+3*8),
        checkpoints=checkpoints,seconds=time.monotonic()-start,RSS_observed_bytes=rss,
        GPU_jobs=0,pretrained_loads=0,limitations='synthetic features/event flags; no recognition, real long input, or GPU speed evidence',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),Path(__file__).with_name('memory_policies.py'),ROOT/'experiments/2b-spatial-time-v1/policy.py')})
    with (OUT/'CPU_long_banks7200.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print({k:v for k,v in result.items() if k not in ('checkpoints','source_sha256')})


if __name__=='__main__':main()
