"""Adapt all-layout and real-pixel tiny-model checks; old preflight stays immutable."""
from pathlib import Path

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
text=(ROOT/'experiments/2b-token-compression-v1/preflight.py').read_text(encoding='utf-8')
text=text.replace('All909', 'All1515').replace('len(layouts)==909','len(layouts)==1515')
text=text.replace("                    else:\n                        assert torch.equal(ids,previous[0])", "                    elif arm=='tome24':\n                        assert torch.equal(ids,previous[0])")
text=text.replace("read_from='compression_three'", "read_from='compression_five'")
text=text.replace("    started = time.monotonic()", "    selection_receipt = check_selection()\n    started = time.monotonic()",1)
text=text.replace("receipt=dict(status='passed'", "receipt=dict(selection_contract=selection_receipt,status='passed'")
addition='''

def check_selection():
    from selection import div_indices, grid_indices
    import math
    # Independent scalar reference, without the vectorized distance update.
    def scalar_ids(x):
        rows=x.double().tolist()
        norm=[max(math.sqrt(sum(v*v for v in row)),1e-12) for row in rows]
        d=[[1-sum(a*b for a,b in zip(rows[i],rows[j]))/(norm[i]*norm[j])
            for j in range(48)] for i in range(48)]
        pick=[max(range(48),key=lambda i:(min(d[i][j] for j in range(48) if j!=i),-i))]
        while len(pick)<24:
            pick.append(max((i for i in range(48) if i not in pick),
                            key=lambda i:(min(d[i][j] for j in pick),-i)))
        return sorted(pick)
    torch.set_num_threads(1)
    torch.manual_seed(930)
    cases=0
    for x in (torch.randn(48,32),torch.zeros(48,32),torch.ones(48,32),torch.eye(48)):
        before=x.clone()
        ids=div_indices(x)
        assert ids.tolist()==scalar_ids(x)
        assert torch.equal(ids,div_indices(x)) and torch.equal(x,before)
        nodes=variants(x,3,6.,7.)
        for arm in ('odd24','grid24','div24'):
            n=nodes[arm]
            original=n.coords[:,1]*8+n.coords[:,2]
            assert len(original.unique())==24 and original.tolist()==sorted(original.tolist())
            assert torch.equal(n.features,x[original]) and n.mass.sum()==24
            assert (n.coords[:,0]==3).all() and n.begin==6. and n.end==7.
            n.features.zero_()  # exported nodes own their values; source untouched
            assert torch.equal(x,before)
        cases+=1
    assert torch.equal(div_indices(torch.zeros(48,2)),torch.arange(24))
    grid=grid_indices()
    assert len(grid)==24 and len(set((grid//8).tolist()))==6
    assert all(int(((grid//8)==r).sum())==4 for r in range(6))
    for x in (torch.full((48,2),float('nan')),torch.ones(47,2)):
        try: div_indices(x)
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')
    a=Archive(pair_cap=1)
    a.append(torch.ones(48,32),0,0.,1.)
    before=a.fingerprint()
    try:a.append(torch.ones(48,32),1,2.,3.)
    except ValueError:pass
    else:raise AssertionError('archive overflow accepted')
    assert before==a.fingerprint()
    a.clear()
    assert all(not ns for ns in a.nodes.values())
    return dict(scalar_reference_cases=cases,zero_tie_indices=list(range(24)),
                original_values_positions_order=True,archive_overflow_atomic=True,
                no_question_argument=True,no_cross_pair_selection_state=True)
'''
text=text.replace("if __name__=='__main__':main()",addition+"\nif __name__=='__main__':main()")
(HERE/'preflight.py').write_text(text,encoding='utf-8',newline='\n')
