from fit import *
torch.set_num_threads(1);packed,hashes=load();done=read(OUT/'completed.json')
assert done['fits']==9 and done['updates']==1620 and done['gpu']==0
record=read(OUT/'inputs.json')
assert record['hashes']==hashes and record['plan_sha']==sha(HERE/'plan.json') and record['script_sha']==sha(HERE/'fit.py')
D=ROOT/'runs/head_data_coverage_20261112_01'
used=[m for p in packed.values() for m in p['meta']]
res=read(D/'STAR_reserve_inputs.json');mlvu=read(D/'MLVU_eval_inputs.json')
for source in (res,mlvu):
    assert not ({m['sha'] for m in used}&{q['media_sha256'] for q in source})
    assert not ({m['source'] for m in used}&{q['source_id'] for q in source})
checks=[]
for name,r in read(OUT/'summary.json').items():
    ck=torch.load(OUT/name/'head.pt',map_location='cpu',weights_only=True);head=Head();head.load_state_dict(ck['state_dict'])
    assert sha(OUT/name/'head.pt')==r['head_sha']
    curve=read(OUT/name/'curve.json');assert min(curve,key=lambda x:x['development_equal_domain_huber'])['step']==ck['selected_step']
    for key,p in packed.items():
        metrics,details=evaluate(head,p,ck['metadata_only']);assert metrics==r['metrics']['/'.join(key)]
    checks.append(name)
extra=[ROOT/'runs/7b_pruned_head_adapt_prepare_20261111_01/fit_sources.json']+[D/f'STAR_{r}_{suffix}.json' for r in ('train','development') for suffix in ('inputs','keys_LOCAL_ONLY')]
save(OUT/'postfit_audit.json',dict(reproduced_heads=checks,source_isolation_with_reserve_and_MLVU=True,
    additional_input_hashes={str(p.relative_to(ROOT)):sha(p) for p in extra},
    heldout_labels_accessed=False,limits=['Manifest isolation audit used IDs/SHA only, not reserve labels or model results','Metadata control excludes semantic question embedding as well as visual descriptors','Additional manifest hashes recorded after fit; original features/score and code hashes verified unchanged']))
print('9 saved heads reproduce all metrics; selected checkpoints match predeclared Huber rule; unused-source isolation passed')
