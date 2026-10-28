"""Paired video-minus-text contributions; labels are only read after all outputs exist."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from text_common import ROOT,OUT,MAIN,read,save,sha,verify,parse_choice,format_ok
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_sember_main import grouped_interval


def lines(p):
    import json
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]


def main(results):
    verify();identity=read(OUT/'local_analysis_identity.json')
    labelpath=ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'
    assert sha(labelpath)==identity['labels_SHA'] and sha(MAIN/'scored_rows_LOCAL_ONLY.json')==identity['baseline_scored_SHA']
    assert sha(ROOT/'scripts/analyze_sember_main.py')==identity['source_interval_code_SHA']
    qs={q['question_id']:q for q in read(OUT/'inputs/requests.json')}
    labels={r['question_id']:r for r in read(labelpath) if r['question_id'] in qs}
    assert len(qs)==len(labels)==478 and all(len(q['options'])==5 for q in qs.values())
    done=read(results/'completed.json');assert done['status']=='complete' and done['answers']==478 and done['vision_calls']==done['video_decodes']==0
    assert done['execution_lock_SHA']==sha(OUT/'inputs/execution_lock.json')
    rows=lines(results/'answers.jsonl');events={}
    for t in lines(results/'tokens.jsonl'):events.setdefault(t['question_id'],[]).append(t['token_id'])
    assert len(rows)==478 and {r['question_id'] for r in rows}==set(qs)
    text={}
    for r in rows:
        qid=r['question_id'];assert r['generated_ids']==events[qid] and 1<=len(r['generated_ids'])<=16
        assert r['branch_released'] and r['visual_tokens']==r['vision_calls']==0
        spec=read(OUT/'inputs/text_specs.json')[qid]
        for key in ('input_sha256','question_suffix_sha256','MCQ_text_sha256'):assert r[key]==spec[key]
        choice=parse_choice(r['text'],5)
        text[qid]={**r,'parsed_choice':choice,'content_correct':choice==labels[qid]['correct_letter'],'format_ok':format_ok(r['text'],5)}
    assert done['generated_tokens']==sum(len(r['generated_ids']) for r in rows)
    assert done['generated_tokens']+done['warmup_generated_tokens']<=7664
    baseline=read(MAIN/'scored_rows_LOCAL_ONLY.json');arms=tuple(dict.fromkeys(r['arm'] for r in baseline))
    assert len(arms)==12 and len(baseline)==5736
    visual={(r['question_id'],r['arm']):r for r in baseline}
    for qid in qs:
        assert all(visual[qid,a]['question_suffix_sha256']==text[qid]['question_suffix_sha256'] for a in arms)
    groups=dict(all=list(qs),ge120s=[q for q in qs if labels[q]['end_gap_proxy_s']>=120],
        ge300s=[q for q in qs if labels[q]['end_gap_proxy_s']>=300])
    assert len(groups['ge120s'])==85 and len(groups['ge300s'])==25
    for cat in sorted({q['question_category'] for q in qs.values()}):groups['type:'+cat]=[k for k,q in qs.items() if q['question_category']==cat]
    scores={};paired={}
    for name,ids in groups.items():
        scores[name]=dict(text_correct=sum(text[q]['content_correct'] for q in ids),total=len(ids),
            uniform_random_expected_correct=len(ids)/5,
            fixed_letter_correct={c:sum(labels[q]['correct_letter']==c for q in ids) for c in 'ABCDE'})
        paired[name]={}
        for a in arms:
            delta={q:int(visual[q,a]['content_correct'])-int(text[q]['content_correct']) for q in qs}
            paired[name][a]=dict(visual_correct=sum(visual[q,a]['content_correct'] for q in ids),
                video_correct_text_wrong=[q for q in ids if delta[q]>0],
                text_correct_video_wrong=[q for q in ids if delta[q]<0],net_video_correct=sum(delta[q] for q in ids),
                same_parsed_choice=sum(text[q]['parsed_choice']==visual[q,a]['parsed_choice'] for q in ids),
                exploratory_source_interval=grouped_interval(ids,qs,delta))
    distributions=dict(gold_formal_only=dict(Counter(labels[q]['correct_letter'] for q in qs)),
        text_only=dict(Counter(text[q]['parsed_choice'] for q in qs)),
        visual={a:dict(Counter(visual[q,a]['parsed_choice'] for q in qs)) for a in arms})
    assert sum(distributions['gold_formal_only'].values())==478
    result=dict(status='complete_text_baseline_comparison',questions=478,sources=128,
        scores=scores,paired_video_minus_text=paired,choice_distributions=distributions,
        text_parsed=sum(r['parsed_choice'] is not None for r in text.values()),
        text_format_ok=sum(r['format_ok'] for r in text.values()),content_and_format_separate=True,
        chance_is_20_percent_not_25=True,random_expected_is_not_a_model_capability_test=True,
        no_claim_model_is_or_is_not_guessing_from_total_score_alone=True,
        posthoc_control_on_opened_dataset=True,not_new_heldout=True,
        no_prompt_tuning_no_question_filtering=True,multiple_comparison_adjustment=False,
        running_resolution_protocol_unchanged=True)
    save(OUT/'text_comparison_analysis.json',result);save(OUT/'text_scored_rows_LOCAL_ONLY.json',list(text.values()))
    print(dict(status=result['status'],scores=scores['all'],paired={a:dict(correct=v['visual_correct'],
        net=v['net_video_correct'],gains=len(v['video_correct_text_wrong']),losses=len(v['text_correct_video_wrong']))
        for a,v in paired['all'].items()},distributions=distributions),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);main(p.parse_args().results)
