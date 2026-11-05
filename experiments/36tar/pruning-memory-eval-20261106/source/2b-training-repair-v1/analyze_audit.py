import argparse, json, statistics
from pathlib import Path
from collections import defaultdict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    rows=[json.loads(x) for x in (a.run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    selection=[json.loads(x) for x in (a.run/'selection.jsonl').read_text(encoding='utf-8').splitlines()]
    metrics=defaultdict(dict)
    for r in rows:metrics[(r['role'],r['view'])].setdefault(r['checkpoint'],[]).append(r)
    report={}
    for group,conditions in metrics.items():
        out={};base={r['id']:r for r in conditions['initial']}
        for name,rs in conditions.items():
            assert set(base)=={r['id'] for r in rs}
            out[name]=dict(answers=len(rs),correct=sum(r['correct'] for r in rs),mean_ce=statistics.mean(r['ce'] for r in rs),gained=sum(r['correct'] and not base[r['id']]['correct'] for r in rs),lost=sum(not r['correct'] and base[r['id']]['correct'] for r in rs),mean_ce_delta=statistics.mean(r['ce']-base[r['id']]['ce'] for r in rs))
        report['/'.join(group)]=out
    sm={}
    for name in dict.fromkeys(r['checkpoint'] for r in selection):
        rs=[r for r in selection if r['checkpoint']==name];ss=[s for r in rs for s in r['stats']]
        sm[name]=dict(sources=len(rs),mean_source_overlap=statistics.mean(r['overlap_initial'] for r in rs),mean_score=statistics.mean(s['mean'] for s in ss),mean_within_pair_std=statistics.mean(s['std'] for s in ss),mean_topk_margin=statistics.mean(s['margin'] for s in ss),max_absolute_score=max(s['absmax'] for s in ss))
    result=dict(metrics=report,selection=sm,scope='fixed diagnostic subset; training probes use the same train rotation; dev shift0 only')
    (a.run/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
