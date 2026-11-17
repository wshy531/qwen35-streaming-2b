"""Merge preserved answers after successful tail; report SB separately."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_prune_state_repair_20261111_01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main(tail,out):
    import torch
    done=read(tail/'completed.json')
    assert done['answers']==1812 and done['state_checks']==150 and done['control_checks']==12
    manifest=read(PREP/'recovery.json');rows={};duplicates=0
    files=[]
    for rel,h in manifest['input_result_files'].items():
        p=ROOT/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files.append(p)
    files.append(tail/'answers.jsonl')
    for p in files:
        for line in p.read_text(encoding='utf8').splitlines():
            r=json.loads(line);key=tuple(r[k] for k in ['id','mode','route','shift'])
            if key in rows:
                prev=rows[key]
                assert r['selected_original']==prev['selected_original'] and r['visual_tokens']==prev['visual_tokens']
                torch.testing.assert_close(torch.tensor(r['option_logits']),torch.tensor(prev['option_logits']),atol=1e-5,rtol=1e-5)
                duplicates+=1
            else:rows[key]=r
    assert len(rows)==3456 and set(Counter(k[0] for k in rows).values())=={12}
    assert duplicates==24
    keys=read(ROOT/'runs/7b_method_transfer_prepare_20261110_01/evaluation_keys_LOCAL_ONLY.json')
    sb=[r for r in rows.values() if r['suite']=='SB150']
    assert len(sb)==1800 and len({r['id'] for r in sb})==150
    result=[]
    for mode in ['full','fixed','content']:
        for route in ['recent8','LS']:
            a=[r for r in sb if r['mode']==mode and r['route']==route]
            assert len(a)==300
            result.append(dict(mode=mode,route=route,correct=sum(r['selected_original']==keys[r['id']] for r in a),
                n=len(a),questions=150,mean_visual_tokens=sum(r['visual_tokens'] for r in a)/len(a)))
    assert not out.exists();out.mkdir()
    (out/'SB150_analysis.json').write_text(json.dumps(dict(tables=result,duplicates_verified=duplicates,
        comparison='same frame-pair coverage, different token budgets; option scoring in two permutations',
        full_history_audit='not completed; LS is not full video history'),indent=2),encoding='utf8')
    print(json.dumps(result))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--tail',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();main(args.tail,args.out)
