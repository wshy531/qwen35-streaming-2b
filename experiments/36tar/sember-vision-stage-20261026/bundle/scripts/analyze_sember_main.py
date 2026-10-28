"""Predeclared all-478 scoring: official content, separate format, paired source CI."""
from collections import defaultdict,Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import ARMS,read,save,sha
from sember_protocol import parse_choice,format_ok
OUT=ROOT/'runs/sember_main_prepare_20261025_01'
SELECTION=ROOT/'runs/sember_prepare_20261025_01/selection'
ALL=(*ARMS,'R_all_history_stream')
CONTROLS={
    'M0_odd54_8':'M0_full108_8','M0_change54_8':'M0_full108_8',
    'M0_odd54_16':'M0_full108_8','M0_change54_16':'M0_full108_8',
    'M2_change54':'M2_full108','A06_change54':'A06_full108','LS_change54':'LS_full108',
    'M2_full108':'M0_full108_8','A06_full108':'M0_full108_8','LS_full108':'M0_full108_8'}
BOOTSTRAP=5000
SEED=20261025


def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def grouped_interval(qids,requests,difference):
    sources=sorted({q['parent_source'] for q in requests.values()})
    index={s:i for i,s in enumerate(sources)};counts=np.zeros(len(sources));deltas=np.zeros(len(sources))
    for qid in qids:
        i=index[requests[qid]['parent_source']];counts[i]+=1;deltas[i]+=difference[qid]
    rng=np.random.default_rng(SEED);sample=rng.integers(0,len(sources),size=(BOOTSTRAP,len(sources)))
    denominator=counts[sample].sum(1);valid=denominator>0
    values=deltas[sample].sum(1)[valid]/denominator[valid]
    if not len(values):return dict(lower=None,upper=None,valid_resamples=0)
    lower,upper=np.quantile(values,[.025,.975])
    return dict(lower=float(lower),upper=float(upper),valid_resamples=int(valid.sum()),
        grouping='original parent video source; participant independence unknown',
        seed=SEED,resamples=BOOTSTRAP,paired=True,multiple_comparison_adjustment=False,
        not_statistical_noninferiority_proof=True)


def freeze():
    qids={q['question_id'] for q in read(OUT/'inputs/requests.json')}
    original=read(SELECTION/'selection_lock.json');assert qids==set(original['formal_question_ids'])
    # Small independent correctness/format and cluster-resampling sanity checks.
    assert parse_choice('Answer: B',5)=='B' and not format_ok('Answer: B',5)
    dummy={str(i):dict(parent_source=str(i//2)) for i in range(8)}
    interval=grouped_interval(list(dummy),dummy,{k:0 for k in dummy})
    assert interval['lower']==interval['upper']==0
    save(OUT/'analysis_lock.json',dict(scorer_SHA=sha(Path(__file__)),
        parser_SHA=sha(ROOT/'experiments/2b-sember-v1/sember_protocol.py'),
        labels_SHA=sha(SELECTION/'labels_LOCAL_ONLY.json'),source_selection_SHA=sha(SELECTION/'selection_lock.json'),
        formal_questions=478,primary_denominator=478,content_and_format_separate=True,
        paired_controls=CONTROLS,source_cluster_bootstrap_resamples=BOOTSTRAP,seed=SEED,
        end_gap_proxy_groups=['all','ge120s','ge300s'],no_output_based_question_filtering=True,
        reference_incorrect_is_not_unanswerable=True,all_outputs_retained=True,
        capability_outputs_observed=False))
    print(dict(status='analysis_plan_frozen',questions=478,methods=12,paired_comparisons=len(CONTROLS)),flush=True)


def main(results):
    lock=read(OUT/'analysis_lock.json');assert sha(Path(__file__))==lock['scorer_SHA']
    assert sha(SELECTION/'labels_LOCAL_ONLY.json')==lock['labels_SHA']
    requests={q['question_id']:q for q in read(OUT/'inputs/requests.json')}
    labels={q['question_id']:q for q in read(SELECTION/'labels_LOCAL_ONLY.json')}
    expected=read(OUT/'inputs/expected_prefixes.json');records={};completed={};startups={}
    for mode,arms in (('bounded',ARMS),('reference',('R_all_history_stream',))):
        completed[mode]=read(results/mode/'completed.json');startups[mode]=read(results/mode/'startup.json')
        assert completed[mode]['status']=='complete' and completed[mode]['vision_encodings']==14678
        rows=lines(results/mode/'answers.jsonl');tokens=lines(results/mode/'tokens.jsonl');token_map=defaultdict(list)
        for t in tokens:token_map[t['question_id'],t['arm']].append(t['token_id'])
        assert len(rows)==478*len(arms)
        assert completed[mode]['answers']==len(rows) and completed[mode]['pretrained_loads']==1
        assert len({(r['question_id'],r['arm']) for r in rows})==len(rows)
        for r in rows:
            qid=r['question_id'];assert qid in requests and r['arm'] in arms
            assert r['read_state_unchanged'] and r['branch_released']
            assert r['generated_ids']==token_map[qid,r['arm']] and 1<=len(r['generated_ids'])<=16
            for field in ('pairs','real_frames','last_committed_pts','submitted_prefix_sha256','true_EOF_tail_padded'):
                assert r['query_prefix'][field]==expected[qid][field]
            choice=parse_choice(r['text'],len(requests[qid]['options']))
            records[qid,r['arm']]={**r,'parsed_choice':choice,'content_correct':choice==labels[qid]['correct_letter'],
                'format_ok':format_ok(r['text'],len(requests[qid]['options'])),
                'multiple_option_letters':len(set(__import__('re').findall(r'\b[A-E]\b',r['text'].upper())))>1}
        assert completed[mode]['generated_tokens']==sum(len(r['generated_ids']) for r in rows)
    assert set(records)=={(q,a) for q in requests for a in ALL}
    assert sum(c['answers']+c['warmup_answers'] for c in completed.values())==5748
    assert sum(c['vision_encodings']+c['warmup_encodings'] for c in completed.values())==29358
    assert sum(c['generated_tokens']+c['warmup_generated_tokens'] for c in completed.values())<=91968
    for qid in requests:
        assert len({records[qid,a]['question_suffix_sha256'] for a in ALL})==1
    def summary(qids,arm):
        rs=[records[qid,arm] for qid in qids]
        return dict(correct=sum(r['content_correct'] for r in rs),total=len(rs),
            parsed=sum(r['parsed_choice'] is not None for r in rs),format_ok=sum(r['format_ok'] for r in rs),
            QA_seconds=sum(r['QA_total_seconds'] for r in rs),audit_seconds=sum(r['state_audit_seconds'] for r in rs))
    groups={'all':list(requests),'ge120s':[q for q in requests if labels[q]['end_gap_proxy_s']>=120],
        'ge300s':[q for q in requests if labels[q]['end_gap_proxy_s']>=300]}
    assert len(groups['ge120s'])==85 and len(groups['ge300s'])==25
    group_scores={name:{a:summary(ids,a) for a in ALL} for name,ids in groups.items()}
    categories=sorted({q['question_category'] for q in requests.values()})
    by_type={cat:{a:summary([qid for qid,q in requests.items() if q['question_category']==cat],a) for a in ALL} for cat in categories}
    by_duration={name:{a:summary([qid for qid,q in requests.items() if (q['duration']<480 if name=='5to8min' else 480<=q['duration']<720 if name=='8to12min' else q['duration']>=720)],a) for a in ALL}
        for name in ('5to8min','8to12min','12plus_min')}
    comparisons={}
    for candidate,control in CONTROLS.items():
        comparisons[candidate]={}
        for name,ids in groups.items():
            diff={q:int(records[q,candidate]['content_correct'])-int(records[q,control]['content_correct']) for q in requests}
            win=[q for q in ids if diff[q]>0];loss=[q for q in ids if diff[q]<0]
            comparisons[candidate][name]=dict(control=control,corrected_ids=win,new_error_ids=loss,
                net_correct=len(win)-len(loss),source_interval=grouped_interval(ids,requests,diff))
    reference_split={a:dict(
        reference_correct_ids=[q for q in requests if records[q,'R_all_history_stream']['content_correct']],
        reference_correct_candidate_wrong=[q for q in requests if records[q,'R_all_history_stream']['content_correct'] and not records[q,a]['content_correct']],
        reference_wrong_candidate_correct=[q for q in requests if not records[q,'R_all_history_stream']['content_correct'] and records[q,a]['content_correct']]) for a in ARMS}
    eligible={}
    for candidate,control in CONTROLS.items():
        if candidate in ('M2_full108','A06_full108','LS_full108'):continue
        nonnegative=all(comparisons[candidate][g]['net_correct']>=0 for g in ('all','ge120s'))
        ratio=group_scores['all'][candidate]['QA_seconds']/group_scores['all'][control]['QA_seconds']
        if candidate.endswith('_8'):gate=nonnegative and ratio<=.85
        else:gate=nonnegative and any(comparisons[candidate][g]['net_correct']>0 for g in ('all','ge120s')) and ratio<=1.05
        eligible[candidate]=dict(enter_independent_cost_validation=gate,QA_ratio=ratio,default_enabled=False,
            no_infinite_parameter_sweep=True)
    result=dict(status='complete_fixed_formal_comparison',questions=478,sources=128,methods=12,
        all_questions_scored=True,content_and_format_separate=True,group_scores=group_scores,by_type=by_type,
        by_duration=by_duration,comparisons=comparisons,reference_split=reference_split,
        validation_gates=eligible,completed=completed,startups=startups,
        no_deletion_of_reference_failures=True,reference_failed_not_intrinsically_unanswerable=True,
        default_changes=False,new_training=False,not_wallclock_1x=True)
    save(OUT/'formal_analysis.json',result)
    save(OUT/'scored_rows_LOCAL_ONLY.json',list(records.values()))
    print(json.dumps(dict(status=result['status'],group_scores=group_scores['all'],validation_gates=eligible),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--results',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze()
    else:
        if args.results is None:parser.error('--results is required')
        main(args.results)
