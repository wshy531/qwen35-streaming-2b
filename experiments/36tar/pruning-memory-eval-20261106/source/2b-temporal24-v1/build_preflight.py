from pathlib import Path
HERE=Path(__file__).parent
ROOT=HERE.parents[1]
text=(ROOT/'experiments/2b-div24-v1/preflight.py').read_text(encoding='utf-8')
text=text.replace('All1515','All909').replace('Archive, variants, context','Archive, context')
start=text.index('    # Independent loop aggregation')
end=text.index('    layouts = []',start)
text=text[:start]+"    contract_cases=selection_receipt['contract_cases']\n"+text[end:]
start=text.index('                    if arm==\'full48\':')
end=text.index('                    layouts.append',start)
text=text[:start]+'''                    assert meta['visual_tokens']==q['pairs']*24
                    assert meta['labels']==[f'<{n.center:.1f} seconds>' for n in archive.nodes['full48']]
'''+text[end:]
text=text.replace('len(layouts)==1515','len(layouts)==909').replace('compression_five','compression_temporal')
text=text.replace('raw48_layouts_equal_existing_reader=303,','same24_visual_count_and_equal_time_labels=909,')
text=text.replace('odd_tome_IDs_and_four_axes_equal=303,','original_selected_coordinates_preserved=True,')
text=text.replace('hash_before = archive.fingerprint()', 'hash_before = archive.fingerprint()\n                selector_before=archive.selector_digest()')
text=text.replace('assert hash_before==archive.fingerprint()', 'assert hash_before==archive.fingerprint() and selector_before==archive.selector_digest()')
text=text[:text.index('\ndef check_selection():')]
text+='''
def check_selection():
    from selection import select,spatial_anchors,odd_indices
    torch.set_num_threads(1)
    x=torch.eye(48); anchors=spatial_anchors(); odd=odd_indices()
    assert len(anchors)==12 and len(anchors.unique())==12
    assert set(anchors.tolist()).issubset(set(odd.tolist()))
    assert all(int(((anchors//8)==r).sum())==2 for r in range(6))
    cold,_=select(x)
    assert all(torch.equal(ids,odd) for ids in cold.values())
    changed=[i for i in range(48) if i not in anchors.tolist()][:12]
    current=x.clone();current[changed]*=-1
    ids,score=select(current,x)
    assert set(changed).issubset(set(ids['change24'].tolist()))
    assert set(ids['mix24'].tolist())==set(changed)|set(anchors.tolist())
    assert all(float(score[i])==2. for i in changed)
    # Independent scalar ranking, no tensor sort or vector normalization.
    import math
    torch.manual_seed(246)
    previous=torch.randn(48,32); current=torch.randn(48,32)
    scalar=[]
    for a,b in zip(current.tolist(),previous.tolist()):
        norm=lambda row:max(math.sqrt(sum(v*v for v in row)),1e-12)
        scalar.append(max(0.,min(2.,1-sum(u*v for u,v in zip(a,b))/(norm(a)*norm(b)))))
    rank=sorted(range(48),key=lambda i:(-scalar[i],i))
    selected,_=select(current,previous)
    assert selected['change24'].tolist()==sorted(rank[:24])
    assert selected['mix24'].tolist()==sorted(anchors.tolist()+[i for i in rank if i not in anchors.tolist()][:12])
    zeros,_=select(torch.zeros(48,32),torch.zeros(48,32))
    assert zeros['change24'].tolist()==list(range(24))
    for name in ('odd24','change24','mix24'):
        a=Archive(pair_cap=2)
        a.append(previous,0,0.,1.);a.append(current,1,2.,3.)
        n=a.nodes[name][-1]; original=n.coords[:,1]*8+n.coords[:,2]
        assert torch.equal(n.features,current[original]) and torch.equal(original,selected[name])
        assert a.status()['selector_bytes']==48*32*4 and a.selector_digest() is not None
        before=(a.fingerprint(),a.selector_digest())
        try:a.append(current,2,4.,5.)
        except ValueError:pass
        else:raise AssertionError('overflow accepted')
        assert before==(a.fingerprint(),a.selector_digest())
        a.clear();assert a.previous is None and all(not ns for ns in a.nodes.values())
    # Selection/retained nodes own their state, no caller tensor aliasing.
    a=Archive();a.append(current,0,0.,1.);digest=a.selector_digest();current.zero_()
    assert digest==a.selector_digest()
    return dict(contract_cases=9,cold_start_equal_odd=True,known_changed_positions_preserved=True,
        scalar_reference=True,static12_preserved=True,zero_vector_ties=True,
        budget_and_original_values=True,previous_pair_owned=True,clear_releases_selector=True,
        failed_append_preserves_state=True)

if __name__=='__main__':main()
'''
(HERE/'preflight.py').write_text(text,encoding='utf-8',newline='\n')
