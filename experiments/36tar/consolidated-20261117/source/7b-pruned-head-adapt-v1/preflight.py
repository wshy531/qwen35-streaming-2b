import argparse,importlib.util
from adapt import *

def main(out):
    verify_adapt();torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('original_micro',ROOT/'experiments/7b-prune-selector-transfer-v1/preflight.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main(out)
    # Exact replay of OLD fitting data demonstrates optimizer and target parity.
    features=torch.load(HERE/'recipe_features.pt',weights_only=True,map_location='cpu')
    fs=[f for f in features if f['role'] in ('train','development')]
    ids={f['id'] for f in fs}
    scores=read(HERE/'recipe_scores.json')
    scores=[r for r in scores if r['id'] in ids]
    dest=out.parent/(out.stem+'_recipe_replay');dest.mkdir(exist_ok=False)
    h=train_head(fs,scores,read(HERE/'fit_sources.json'),dest);ref=load_head()
    # Independent OLD implementation on the SAME CPU. Cross-OS BLAS training
    # accumulates small roundoff; do not conflate that with changed recipe.
    old_source=ROOT/'experiments/7b-method-recovery-v1/fit_partial.py'
    source=old_source.read_text(encoding='utf8').split('    # Only now open evaluation answers;')[0]
    assert source.count('==(144,32,177)')==1
    assert source.count("[('visible',False),('blind',True)]")==1
    source=source.replace('==(144,32,177)','==(144,32,0)').replace("[('visible',False),('blind',True)]","[('visible',False)]")
    source+="    return heads['visible']\n"
    namespace={'__file__':str(old_source)};exec(compile(source,str(old_source),'exec'),namespace)
    olddest=out.parent/(out.stem+'_original_recipe');olddest.mkdir(exist_ok=False)
    torch.save(fs,olddest/'features.pt')
    (olddest/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores),encoding='utf8')
    original=namespace['fit_and_evaluate'](olddest)
    for name,value in h.state_dict().items():torch.testing.assert_close(value,original.state_dict()[name],rtol=0,atol=0)
    cross_os=max(float((v-ref.state_dict()[n]).abs().max()) for n,v in h.state_dict().items())
    decisions_equal=0;max_prediction_delta=0.
    with torch.no_grad():
        for f in fs:
            choices,mask=choices_mask(int(f['node_mask'].sum()))
            inp=[f[k][None] for k in FIELDS]+[mask]
            x,y=h(*inp)[0],ref(*inp)[0]
            decisions_equal+=choose(x,choices)==choose(y,choices)
            max_prediction_delta=max(max_prediction_delta,float((x-y).abs().max()))
    # Test actual pruned descriptors, NULL, all subset sizes, and update gradients.
    torch.manual_seed(711111);m=Memory('content');checks=[]
    for i in range(12):
        m.append(torch.randn(70,3584),torch.arange(1,140,2),i,2*i,2*i+1)
        recent,history=m.parts();f=descriptor_features(torch.randn(3584),recent,history)
        assert torch.equal(f['metadata'][f['node_mask'],5],torch.full((len(history),),.5))
        nodes,choice,_=select_nodes(h,f['query'],recent,history)
        assert sum(len(n.features) for n in nodes)<=560
        checks.append(dict(pair=i,nodes=len(history),choice=choice))
    result=read(out);result.update(adapt_passed=True,old_recipe_bitwise_equal=True,
        cross_platform_max_parameter_abs=cross_os,cross_platform_max_prediction_abs=max_prediction_delta,
        cross_platform_same_decisions=decisions_equal,recipe_reference_sha=sha(old_source),
        pruned_descriptor_checks=checks,lock_sha=sha(HERE/'lock.json'),train_questions=144,eval_questions=210)
    save(out,result);print(json.dumps(dict(adapt_passed=True,old_recipe_bitwise_equal=True,lock_sha=result['lock_sha'])))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
