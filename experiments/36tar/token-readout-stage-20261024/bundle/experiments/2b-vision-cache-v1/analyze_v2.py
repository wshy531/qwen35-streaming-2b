"""Prespecified content score, separate format and visual-component costs."""
import argparse
import importlib.util
import json
from pathlib import Path
from common import ROOT,OUT,ARMS,read,save,sha

spec=importlib.util.spec_from_file_location('fixed_content_parser',ROOT/'experiments/2b-standalone-cost-v1/analyze.py')
parser_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser_module)


def main(result):
    complete=read(result/'completed.json')
    assert complete['status']=='complete' and complete['formal_answers']==606 and complete['formal_visual_calls']==7962
    questions={q['id']:q for q in read(OUT/'inputs/questions.json')}
    labels={q['id']:q for q in read(OUT/'labels_LOCAL_ONLY.json')}
    rows=[json.loads(line) for line in (result/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    predictions={(r['id'],r['arm']):r for r in rows}
    assert len(rows)==len(predictions)==606
    details=[]
    for uid,q in questions.items():
        for arm in ARMS:
            p=predictions[uid,arm]
            parsed=parser_module.parse(p['text'],q['options'],status=p['status'])
            assert p['banks_unchanged'] and p['branch_released']
            details.append(dict(id=uid,arm=arm,group=q['group'],cohort=q['cohort'],text=p['text'],
                answer_correct=parsed['choice']==labels[uid]['answer_index'],**parsed))
    lookup={(r['id'],r['arm']):r for r in details}
    sets={name:[uid for uid,q in questions.items() if q['cohort']==name] for name in ('natural171','T98')}
    sets['primary269']=sets['natural171']+sets['T98']
    sets['long_original28']=[uid for uid,q in questions.items() if q['cohort']=='stream34' and uid.startswith('ovo:')]
    sets['long_supported16']=[uid for uid in sets['long_original28'] if labels[uid]['verdict']=='supported']
    sets['long_manual6']=[uid for uid,q in questions.items() if q['cohort']=='stream34' and not uid.startswith('ovo:')]
    assert len(sets['long_original28'])==28 and len(sets['long_supported16'])==16 and len(sets['long_manual6'])==6
    for cohort in ('natural171','T98','primary269'):
        for typ in ('F','T','C'):
            sets[cohort+'_'+typ]=[uid for uid in sets[cohort] if labels[uid]['stratum']==typ]
    tables={}
    for name,ids in sets.items():
        tables[name]=dict(denominator=len(ids),scores={a:dict(
            correct=sum(lookup[u,a]['answer_correct'] for u in ids),
            format_compliant=sum(lookup[u,a]['format_compliant'] for u in ids),
            unjudgeable=sum(lookup[u,a]['unjudgeable'] for u in ids)) for a in ARMS},
            corrections=[u for u in ids if lookup[u,'cache25_8']['answer_correct'] and not lookup[u,'full48_8']['answer_correct']],
            new_losses=[u for u in ids if lookup[u,'full48_8']['answer_correct'] and not lookup[u,'cache25_8']['answer_correct']])
    historical=read(OUT/'historical_references_LOCAL_ONLY.json')['records']
    regressions={key:[r['id'] for r in historical if predictions[r['id'],'full48_8'][key]!=r[key]]
        for key in ('generated_ids','input_ids_sha256','node_digest')}
    visual={}
    for name,subset in [('all112',complete['profiles']),('long6',[p for p in complete['profiles'] if p['cohort']=='stream34'])]:
        times={a:sum(p['per_arm_encoding'][a]['vision_seconds'] for p in subset) for a in ARMS}
        visual[name]=dict(visual_seconds=times,candidate_lower=times['cache25_8']<times['full48_8'],
            reduction=1-times['cache25_8']/times['full48_8'])
    quality_gate=all(tables[s]['scores']['cache25_8']['correct']>=tables[s]['scores']['full48_8']['correct']
                     for s in ('primary269','long_supported16'))
    report=dict(status='analyzed',answers=606,tables=tables,details=details,baseline_regressions=regressions,
        visual_component_cost=visual,quality_gate=quality_gate,
        candidate_meets_prespecified_gates=quality_gate and all(v['candidate_lower'] for v in visual.values()) and not any(regressions.values()),
        reference_bytes=75497472,overall_two_arm_GPU_peak=complete['overall_two_arm_peak_GPU_allocated_bytes'],
        limitations=['opened development sets, not unseen evidence','no independent full-pipeline or edge/1x timing',
            'two banks coexist, so overall GPU peak is not an individual-method peak',
            'passing gates only retains a candidate; it does not prove stable improvement',
            'format scored separately from content; ambiguous/manual long items separate'],
        source_labels_SHA256=sha(OUT/'labels_LOCAL_ONLY.json'))
    save(OUT/'GPU_results_analysis_v2.json',report)
    print(dict(answers=606,quality_gate=quality_gate,visual_cost=visual,regressions=regressions))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    main(parser.parse_args().results)
