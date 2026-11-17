from fit import *
torch.set_num_threads(1)
packed,hashes=load()
torch.manual_seed(17);head=Head()
for key,p in packed.items():
    assert torch.equal(p['target'][:,0],torch.zeros(len(p['meta'])))
    assert not bool(p['mx'][1].any()) and not bool(p['mx'][2].any())
    assert not bool(p['mx'][0][:,4:].any())
    for mode in ('x','mx'):
        x=[v[:3] for v in p[mode]];pred=head(*x)
        loss=objective(pred,p['target'][:3],p['valid'][:3],torch.ones(len(pred)))
        assert torch.isfinite(loss);head.zero_grad();loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in head.parameters())
    print(key,len(p['meta']),len({m['source'] for m in p['meta']}))
print('real features, gradients, null target, metadata blindness and source isolation passed')
