"""Post-fit train/development diagnosis, independent of evaluation answers."""
import argparse
from adapt import *

def main(out):
    torch.set_num_threads(1)
    fs=torch.load(out/'features.pt',weights_only=True,map_location='cpu')
    scores=[json.loads(l) for l in (out/'scores.jsonl').read_text().splitlines()]
    rows={(r['id'],r['subset'],r['shift']):r for r in scores}
    groups=read(HERE/'fit_sources.json')
    gold={q['id']:q['answer_index'] for g in groups for q in g['questions']}
    assert len(fs)==176 and all(f['id'] in gold for f in fs)
    old=load_head();new=Head()
    ck=torch.load(out/'head-visible.pt',weights_only=True,map_location='cpu')
    new.load_state_dict(ck['state_dict']);new.eval().requires_grad_(False)
    result={};details=[]
    with torch.no_grad():
        for role in ('train','development'):
            subset=[f for f in fs if f['role']==role];acc={};benefit={};counts={};regret={}
            opportunity=dict(any_correctness_gain_questions=0,multiple_nodes_needed_for_best_correctness=0,
                best_probability_uses_multiple_and_gain_ge_001=0,probability_gain_ge_001_questions=0,
                all_subsets_wrong_questions=0,selected_NULL_but_LS_better_questions=0,
                oracle_correct_total=0,best_zero_or_one_correct_total=0)
            for f in subset:
                choices,mask=choices_mask(int(f['node_mask'].sum()))
                assert choices==f['choices']
                probs=[];correct=[]
                for j in range(len(choices)):
                    pp=[];cc=[]
                    for shift in (0,2):
                        r=rows[f['id'],j,shift];v=torch.tensor(r['option_logits'])
                        pp.append(float(v.softmax(-1)[(gold[f['id']]-shift)%len(v)]))
                        cc.append(r['selected_original']==gold[f['id']])
                    probs.append(sum(pp)/2);correct.append(sum(cc))
                gains=torch.tensor(probs)-probs[0]
                selected={'recent':0,'LS':len(choices)-1,'oracle_probability':choose(gains,choices)}
                for name,head in [('old_head',old),('adapted',new)]:
                    pred=head(*(f[k][None] for k in FIELDS),mask)[0]
                    selected[name]=choose(pred,choices)
                selected['adapted_same_count_earliest']=choices.index(list(range(len(choices[selected['adapted']]))))
                best=max(correct);best01=max(c for c,choice in zip(correct,choices) if len(choice)<=1)
                opportunity['oracle_correct_total']+=best;opportunity['best_zero_or_one_correct_total']+=best01
                opportunity['any_correctness_gain_questions']+=best>correct[0]
                opportunity['multiple_nodes_needed_for_best_correctness']+=best>best01
                opportunity['probability_gain_ge_001_questions']+=float(gains.max())>=.01
                opportunity['best_probability_uses_multiple_and_gain_ge_001']+=(len(choices[selected['oracle_probability']])>1 and float(gains.max())>=.01)
                opportunity['all_subsets_wrong_questions']+=best==0
                opportunity['selected_NULL_but_LS_better_questions']+=(selected['adapted']==0 and correct[-1]>correct[0])
                for name,j in selected.items():
                    acc[name]=acc.get(name,0)+correct[j]
                    benefit[name]=benefit.get(name,0)+float(gains[j])/len(subset)
                    regret[name]=regret.get(name,0)+float(gains.max()-gains[j])/len(subset)
                    counts.setdefault(name,[]).append(len(choices[j]))
                details.append(dict(id=f['id'],role=role,actual_gains=gains.tolist(),choices=choices,selected=selected))
            from collections import Counter
            result[role]=dict(questions=len(subset),n=2*len(subset),content_correct=acc,
                mean_probability_gain=benefit,mean_probability_regret=regret,
                selected_node_counts={k:dict(Counter(v)) for k,v in counts.items()},opportunity=opportunity)
    save(out/'supervision_analysis.json',dict(tables=result,details=details,
        limits=['Training metrics are in-sample.','Oracle uses gold after the fact, not an automatic method.',
        'Probability gain and correctness are distinct targets.','Development group does not select epochs.']))
    print(json.dumps(result))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
