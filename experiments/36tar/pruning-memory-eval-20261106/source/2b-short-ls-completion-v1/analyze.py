"""Analyze completed LS-only short panel against the frozen full108 LS bank."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def mapping(rows):
    result={(r['id'],r['shift']):r for r in rows}
    assert len(rows)==len(result)==690
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    complete=read(a.run/'completed.json')
    assert complete['status']=='completed' and complete['answers']==2760 and complete['regressions']==2070
    baseline=ROOT/'runs/history-utility-short-220532'
    assert read(baseline/'completed.json')['status']=='completed'
    full=mapping([r for r in lines(baseline/'answers.jsonl') if r['phase']=='fit' and r['subset']==15])
    rows=lines(a.run/'answers.jsonl');results={}
    for arm in ('initial','old_CE','guarded','fixed12'):
        obs=mapping([r for r in rows if r['phase']=='evaluation' and r['arm']==arm])
        assert obs.keys()==full.keys()
        correct=sum(r['correct'] for r in obs.values())
        types={}
        for r in obs.values():
            v=types.setdefault(r['type'][0],dict(correct=0,answers=0))
            v['correct']+=int(r['correct']);v['answers']+=1
        results[arm]=dict(correct=correct,answers=690,accuracy=correct/690,
            gain_vs_full108=sum(not full[k]['correct'] and obs[k]['correct'] for k in full),
            loss_vs_full108=sum(full[k]['correct'] and not obs[k]['correct'] for k in full),by_type=types)
    report=dict(questions=345,sources=40,option_shifts=[0,2],
        full108_LS=dict(correct=sum(r['correct'] for r in full.values()),answers=690),methods=results,
        limits=['opened development, not unseen final test',
                'same LS864 budget; full108 and prune54 have different temporal coverage',
                'option classification, not explanation judging or independent cost measurement'])
    (a.run/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
