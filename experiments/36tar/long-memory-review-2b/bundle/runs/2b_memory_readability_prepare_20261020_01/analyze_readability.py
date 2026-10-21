"""Local strict scoring, same parser as earlier runs; retain all six cases."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
from score_memory_eval import parse_choice


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main(results):
    labels={r['id']:r for r in read(OUT/'labels6_LOCAL_ONLY.json')}
    rows={r['id']:r for r in read(OUT/'requests6_LOCAL.json')}
    refs={r['id']:r for r in read(OUT/'bank_reproduction_reference_LOCAL_ONLY.json')}
    arms=['A_raw_old','B_compressed_old','C_raw_old_recent','D_compressed_old_recent','E_full_pairwise','F_bank_reproduction']
    predictions={}
    for line in (results/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        p=json.loads(line);key=(p['id'],p['arm'])
        assert key not in predictions and key[0] in rows and key[1] in arms
        predictions[key]=p
    assert len(predictions)<=36
    cases=[];counts={arm:0 for arm in arms};formats={arm:0 for arm in arms}
    for ident,row in rows.items():
        items={}
        for arm in arms:
            p=predictions.get((ident,arm),dict(status='missing',text=None,generated_ids=[]))
            choice=parse_choice(p.get('text'),len(row['options'])) if p.get('status')=='ok' else None
            correct=choice is not None and choice==labels[ident]['answer_index']
            items[arm]=dict(**p,predicted_index=choice,correct=correct)
            counts[arm]+=int(correct);formats[arm]+=int(choice is None)
        right=lambda a:items[a]['correct']
        cases.append(dict(id=ident,question=row['question'],options=row['options'],answer_index=labels[ident]['answer_index'],
            arms=items,flags=dict(
                A_correct_B_wrong=right(arms[0]) and not right(arms[1]),
                B_correct_D_wrong=right(arms[1]) and not right(arms[3]),
                C_correct_D_wrong=right(arms[2]) and not right(arms[3]),
                D_correct_F_wrong=right(arms[3]) and not right(arms[5]),
                A_wrong_E_correct=not right(arms[0]) and right(arms[4]),
                E_wrong_native_previous_correct=not right(arms[4])),
            F_generated_IDs_reproduce_previous=items[arms[5]]['generated_ids']==refs[ident]['generated_ids'],
            selected_source_ranges=row['selected_source_ranges'],
            raw_visual_tokens=48*len(row['raw_selected_pair_indices']),
            previous_F_text=refs[ident]['text']))
    reproduced=sum(c['F_generated_IDs_reproduce_previous'] for c in cases)
    report=dict(denominator_per_arm=6,total_predictions=len(predictions),strict_correct=counts,invalid_or_missing=formats,
        F_exact_reproduced=reproduced,F_reproduction_complete=reproduced==6,cases=cases,
        interpretation_limits=[
            'six already-open single-assistant development questions, no general accuracy claim',
            'oracle chosen old source ranges, not an automatic retrieval method',
            'A/B changes compressed values and surviving-coordinate layout together',
            'A_wrong is failure of selected uncompressed context, not necessarily visual recognition failure',
            'E/native differences include pairwise visual and readout protocol; not isolated compression',
            'F must reproduce before treating the old/new comparison as established',
            'diagnostic raw history consumes extra memory and is not a bounded runtime candidate',
            'strict format errors kept as errors; raw output retained without relaxed rescoring'])
    with (OUT/'readability_strict_results.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(correct=counts,F_reproduced=reproduced,answers=len(predictions)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);main(p.parse_args().results)
