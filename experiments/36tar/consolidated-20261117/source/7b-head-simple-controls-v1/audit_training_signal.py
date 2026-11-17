"""Inspect already frozen train/dev labels against latest-node baseline; no training."""
import collections,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    groups=read(ROOT/'runs/7b_pruned_head_adapt_prepare_20261111_01/fit_sources.json')
    rows=[json.loads(x) for x in (ROOT/'runs/7b-pruned-head-adapt-223422/snapshot-01/scores.jsonl').read_text(encoding='utf8').splitlines()]
    data=collections.defaultdict(dict)
    for r in rows:data[r['id']][r['subset'],r['shift']]=r
    output={}
    for role in ('train','development'):
        entries=[]
        for g in groups:
            if g['role']!=role:continue
            for q in g['questions']:
                scores=data[q['id']];n_subsets=len(scores)//2;n=n_subsets.bit_length()-1
                assert 2**n==n_subsets
                choices=[list(c) for k in range(n+1) for c in itertools.combinations(range(n),k)]
                last=choices.index([n-1]) if n else 0
                p=[];c=[]
                for j in range(n_subsets):
                    pp=[];cc=[]
                    for s in (0,2):
                        row=scores[j,s];v=np.array(row['option_logits']);z=np.exp(v-v.max());z/=z.sum()
                        pp.append(float(z[(q['answer_index']-s)%len(v)]))
                        cc.append(int(row['selected_original']==q['answer_index']))
                    p.append(float(np.mean(pp)));c.append(sum(cc))
                entries.append(dict(id=q['id'],recent_correct=c[0],last_correct=c[last],LS_correct=c[-1],
                    oracle_correct=max(c),oracle_gain=max(c)-c[last],
                    probability_room=max(p)-p[last],argmax_probability_subset=choices[int(np.argmax(p))],
                    earliest_correct=c[1] if n else c[0],all_wrong=max(c)==0))
        output[role]=dict(questions=len(entries),recent_correct=sum(e['recent_correct'] for e in entries),
            last_correct=sum(e['last_correct'] for e in entries),LS_correct=sum(e['LS_correct'] for e in entries),
            oracle_one_subset_per_question_correct=sum(e['oracle_correct'] for e in entries),
            questions_with_correctness_room_over_last=sum(e['oracle_gain']>0 for e in entries),
            questions_with_probability_room_over_001=sum(e['probability_room']>.01 for e in entries),
            all_subsets_wrong_questions=sum(e['all_wrong'] for e in entries),rows=entries)
    out=ROOT/'runs/7b-head-simple-223656/training_signal.json'
    out.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:{n:v for n,v in x.items() if n!='rows'} for k,x in output.items()}))
if __name__=='__main__':main()
