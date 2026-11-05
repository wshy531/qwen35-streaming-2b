"""Check completeness, references, repeats, and the disclosed cost after a batch."""
import argparse,json,math
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    done=read(a.run/'completed.json');budget=read(a.run/'budget.json')
    tasks=read(a.inputs/'tasks.json');cases=read(a.inputs/'cases_LOCAL_ONLY.json');rep=read(a.inputs/'repeats_LOCAL_ONLY.json');cal=read(a.inputs/'calibrations_LOCAL_ONLY.json')
    expected={x['opaque_id'] for x in cases+rep+cal}|{x['task_id'] for x in tasks}
    raw=[json.loads(l) for l in (a.run/'responses.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(raw)==len(expected)==done['requests']==budget['requests']
    assert {r['id'] for r in raw}==expected and budget['pending'] is None
    cost=sum(r['usage']['prompt_tokens']*2e-6+r['usage']['completion_tokens']*8e-6 for r in raw)
    assert math.isclose(cost,budget['peak_upper_cny'],abs_tol=1e-9) and cost<=budget['cap_cny']
    refs={x['task_id']:{f['frame_id'] for f in x['frames']} for x in tasks}
    invalid=[];pending=[]
    judged=read(a.run/'judgments.json')
    assert len(judged)==len(cases)+len(rep)
    assert {r['opaque_id'] for r in judged}=={r['opaque_id'] for r in cases+rep}
    for row in judged:
        j=row['judgment']
        if j.get('processing_status')!='parsed':pending.append(row['opaque_id'])
        for c in j.get('checks',[]):
            if any(x not in refs[row['task_id']] for x in c.get('references',[])):invalid.append(row['opaque_id'])
    assert not invalid,invalid
    result=dict(requests=len(raw),judgments=len(judged),unique_ids_verified=True,reference_ids_verified=True,
                pending_parser=pending,peak_price_cost_recomputed_cny=cost,ledger_no_pending=True,
                API=0,GPU=0,not_independent_content_review=True)
    (a.run/'integrity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
