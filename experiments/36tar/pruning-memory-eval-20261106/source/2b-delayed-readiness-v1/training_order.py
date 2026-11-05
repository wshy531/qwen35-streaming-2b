"""Same sample multiset and optimizer-step count for curriculum versus mixture.

Each update uses two separately recomputed examples with loss scale 0.5;
clip and optimizer.step happen once after BOTH backwards, never in between.
"""
from collections import Counter


def schedules(short_ids, delayed_ids):
    n=len(short_ids)
    if n<2 or n%2 or len(delayed_ids)!=n:
        raise ValueError('need equal nonzero even exposure counts; never pad silently')
    short=[dict(kind='short',id=i,loss_scale=.5) for i in short_ids]
    delayed=[dict(kind='delayed',id=i,loss_scale=.5) for i in delayed_ids]
    result=dict(
        short_then_delayed=[short[i:i+2] for i in range(0,n,2)]+[delayed[i:i+2] for i in range(0,n,2)],
        mixed=[[short[i],delayed[i]] for i in range(n)])
    expected=Counter((r['kind'],r['id']) for r in short+delayed)
    for batches in result.values():
        assert len(batches)==n and all(len(b)==2 for b in batches)
        assert Counter((r['kind'],r['id']) for b in batches for r in b)==expected
        assert all(sum(r['loss_scale'] for r in b)==1 for b in batches)
    return result


if __name__=='__main__':
    import json
    from pathlib import Path
    checks=0
    for n in (2,4,10,100,200):
        data=schedules([f's{i}' for i in range(n)],[f'l{i}' for i in range(n)])
        assert all(r['kind']=='short' for b in data['short_then_delayed'][:n//2] for r in b)
        assert all({r['kind'] for r in b}=={'short','delayed'} for b in data['mixed'])
        checks+=1
    for short,long in [([],[]),(['a'],['b']),(['a','b'],['c'])]:
        try:schedules(short,long)
        except ValueError:checks+=1
        else:raise AssertionError('invalid schedule accepted')
    out=Path(__file__).resolve().parents[2]/'runs/delayed_training_candidates_20261104_01/CPU_training_order.json'
    with out.open('x',encoding='utf-8') as f:json.dump(dict(checks=checks,status='passed',
        production_samples_frozen=False,updates=0,GPU=0,example=schedules(['s0','s1','s2','s3'],['l0','l1','l2','l3'])),f,indent=2)
    print(json.dumps(dict(checks=checks,status='passed',training_run=False)))
