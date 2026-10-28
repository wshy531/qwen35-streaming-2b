"""Local-only diagnosis: never choose questions or tune prompts from the results."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
from audit_common import ROOT,OUT,read,save
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_protocol import parse_choice,format_ok


def jsonl(path):
    import json
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def main(results):
    import numpy as np
    identity=read(OUT/'local_analysis_identity.json')
    for rel,digest in identity['frozen_inputs'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    assert read(results/'completed.json')['status']=='complete'
    qs={q['question_id']:q for q in read(OUT/'inputs/requests.json')}
    labels={q['question_id']:q for q in read(ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json')}
    conditions=read(OUT/'inputs/design.json')['conditions']
    rows={}
    for condition in conditions:
        items=jsonl(results/condition/'answers.jsonl')
        assert len(items)==24 and {r['question_id'] for r in items}==set(qs)
        assert read(results/condition/'completed.json')['status']=='complete'
        rows[condition]={r['question_id']:r for r in items}
        for r in items:
            p=results/condition/r['first_logits_file']
            assert p.parent==results/condition
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['first_logits_sha256']
    reference=jsonl(ROOT/'runs/sember_main_prepare_20261025_01/GPU_results_218526/reference/answers.jsonl')
    rows['C_existing_full_stream']={r['question_id']:r for r in reference if r['question_id'] in qs}
    assert len(rows['C_existing_full_stream'])==24
    for arm,items in rows.items():
        for qid,r in items.items():
            r['choice']=parse_choice(r['text'],len(qs[qid]['options']))
            r['correct']=r['choice']==labels[qid]['correct_letter']
            r['format_ok']=format_ok(r['text'],len(qs[qid]['options']))
    scores={arm:dict(correct=sum(r['correct'] for r in items.values()),questions=24,
        parsed=sum(r['choice'] is not None for r in items.values()),
        format_ok=sum(r['format_ok'] for r in items.values()),
        prediction_counts=dict(Counter(r['choice'] for r in items.values()))) for arm,items in rows.items()}
    paired={}
    for a,b in [(conditions[0],conditions[1]),(conditions[1],conditions[2]),(conditions[2],'C_existing_full_stream')]:
        paired[a+'_vs_'+b]=dict(
            corrected=[q for q in qs if rows[a][q]['correct'] and not rows[b][q]['correct']],
            new_errors=[q for q in qs if not rows[a][q]['correct'] and rows[b][q]['correct']],
            identical_generated_ids=sum(rows[a][q]['generated_ids']==rows[b][q]['generated_ids'] for q in qs))
    version_errors={q:float(np.abs(np.load(results/conditions[1]/(q+'.first_logits.npy'))-
        np.load(results/conditions[2]/(q+'.first_logits.npy'))).max()) for q in qs}
    for q in qs:
        for key in ('input_ids_sha256','pixels_sha256','grid','input_tokens','visual_tokens'):
            assert rows[conditions[1]][q]['input'][key]==rows[conditions[2]][q]['input'][key],(q,key)
    save(OUT/'profile_analysis.json',dict(status='complete',scores=scores,paired=paired,
        B540_vs_B530_first_logits_max_abs=version_errors,
        profile_is_not_full478_accuracy_or_independent_generalization=True,
        no_selection_or_prompt_tuning=True,format_separate=True))
    print(scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    main(p.parse_args().results)
