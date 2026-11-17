import argparse,importlib.util
from combination import *

def main(out):
    verify_all();torch.set_num_threads(1)
    questions=read(HERE/'questions.json');plan=read(HERE/'plan.json')
    assert len(questions)==210 and {q['suite'] for q in questions}=={'SB150','short60'}
    refs=read(HERE/'references.json');keys=[tuple(r[k] for k in ('id','mode','route','shift')) for r in refs]
    assert len(set(keys))==len(keys)==1860
    for r in refs:check_reference(r,r)
    bad=dict(refs[0]);bad['selected_original']=-1
    try:check_reference(bad,refs[0])
    except AssertionError:pass
    else:raise AssertionError('corrupt reference accepted')
    # Reuse real official tiny forward/positions for unchanged pruning operators.
    spec=importlib.util.spec_from_file_location('original_preflight',ROOT/'experiments/7b-prune-selector-transfer-v1/preflight.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main(out)
    h=load_head();before={n:x.clone() for n,x in h.state_dict().items()};tests=[]
    torch.manual_seed(7111)
    for mode in ['full','content']:
        memory=Memory(mode);query=torch.randn(3584)
        ids=torch.arange(140) if mode=='full' else torch.arange(1,140,2)
        for i in range(12):
            memory.append(torch.randn(len(ids),3584),ids,i,2.*i,2.*i+1)
            recent,history=memory.parts();snapshot=digest(memory.bank.nodes())
            nodes,choice,detail=select_nodes(h,query,recent,history)
            assert all(0<=j<len(history) for j in choice)
            assert nodes[-len(recent):]==recent
            assert digest(memory.bank.nodes())==snapshot and memory.archive is None
            assert sum(len(n.features) for n in nodes)<= (1120 if mode=='full' else 560)
            tests.append(dict(mode=mode,pair=i,history=len(history),choice=choice,tokens=sum(len(n.features) for n in nodes)))
    assert all(torch.equal(before[n],x) for n,x in h.state_dict().items()) and all(x.grad is None for x in h.parameters())
    result=read(out);result.update(combination_passed=True,frozen_head_checks=tests,questions=210,references=1860,lock_sha=sha(HERE/'lock.json'))
    save(out,result);print(json.dumps(dict(combination_passed=True,frozen_head_checks=len(tests),questions=210)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
