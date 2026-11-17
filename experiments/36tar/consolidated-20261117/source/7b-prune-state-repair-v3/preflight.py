import sys,ast,copy,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-prune-selector-transfer-v1'))
from common import *
from recovery_common import REC,specification,check_control,verify_recovery
from state_contract import compare_state,FIELD

def main(out):
    verify();verify_recovery();s=specification('prune')
    assert len(s['missing_ids'])==len(set(s['missing_ids']))==150
    assert s['control_id'] not in s['missing_ids']
    assert {q['id'] for q in s['questions']}==set(s['missing_ids'])|{s['control_id']}
    assert all(q['suite']=='SB150' for q in s['questions'] if q['id']!=s['control_id'])
    for row in s['references']:check_control(row,s['references'],'prune')
    trace=read(PREP/'traces.json');state_rows=0;negative=0
    for q in s['questions']:
        ref=trace[q['suite']+'|'+q['source_id']].get('states',{}).get(q['id'])
        if ref is None:continue
        actual=dict(ref);actual[FIELD]=280*392*3
        compare_state(actual,ref);state_rows+=1
        for k in actual:
            bad=copy.deepcopy(actual)
            bad[k]=not bad[k] if isinstance(bad[k],bool) else bad[k]+1
            try:compare_state(bad,ref)
            except AssertionError:negative+=1
            else:raise AssertionError(('corrupt causal state accepted',q['id'],k))
    assert state_rows==150 and negative==1500
    script=Path(__file__).parent/'run_prune.sh';assert b'\r' not in script.read_bytes()
    ast.parse((Path(__file__).parent/'run_prune_tail.py').read_text(encoding='utf8'))
    modspec=importlib.util.spec_from_file_location('original_preflight',ROOT/'experiments/7b-prune-selector-transfer-v1/preflight.py')
    m=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(m);m.main(out)
    r=read(out);r.update(recovery_contracts=True,lock_sha=sha(REC/'lock.json'),
        state_rows=state_rows,rejected_corrupt_states=negative,expected_answers=1812)
    save(out,r)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
