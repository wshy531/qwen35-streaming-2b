"""Content-first paired analysis; exact prior replay, isolated cost and all failures retained."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
from quality_common import ROOT,OUT,ARMS,read,save,sha,verify,lines
from sember_protocol import parse_choice,format_ok
from quality_statistics import grouped_interval


def main(results):
    protocol=verify();assert read(results/'completed.json')['status']=='complete'
    labelpath=ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'
    assert sha(labelpath)==read(OUT/'local_analysis_identity.json')['labels_SHA']
    qs={q['question_id']:q for q in read(OUT/'inputs/requests.json')}
    labels={q['question_id']:q for q in read(labelpath)};records={};completed=[]
    for task in read(OUT/'inputs/tasks.json'):
        base=results/task['name'];c=read(base/'completed.json');assert c['status']=='complete';completed.append(c)
        rows=lines(base/'answers.jsonl');events=defaultdict(list)
        for e in lines(base/'tokens.jsonl'):events[e['question_id'],e['arm']].append(e['token_id'])
        ids={q for q,v in qs.items() if task['videos'] is None or v['video_id'] in task['videos']}
        assert len(rows)==len(ids)*len(task['arms'])==c['answers']
        assert {(r['question_id'],r['arm']) for r in rows}=={(q,a) for q in ids for a in task['arms']}
        for r in rows:
            assert r['generated_ids']==events[r['question_id'],r['arm']] and 1<=len(r['generated_ids'])<=16
            assert r['read_state_unchanged'] and r['branch_released']
            if task['kind']=='quality':records[r['question_id'],r['arm']]=r
            else:
                old=records[r['question_id'],r['arm']]
                for k in ('generated_ids','text','visual_tokens','bounded_context_ids_sha256','question_suffix_sha256'):assert r[k]==old[k]
    assert set(records)=={(q,a) for q in qs for a in ARMS}
    assert sum(c['answers']+c['warmup_answers'] for c in completed)==2160
    assert sum(c['vision_encodings']+c['warmup_encodings'] for c in completed)==37014
    for (q,a),r in records.items():
        r.update(parsed_choice=parse_choice(r['text'],len(qs[q]['options'])),
            format_ok=format_ok(r['text'],len(qs[q]['options'])))
        r['correct']=r['parsed_choice']==labels[q]['correct_letter']
    groups=dict(all=list(qs),ge120s=[q for q in qs if labels[q]['end_gap_proxy_s']>=120],
        ge300s=[q for q in qs if labels[q]['end_gap_proxy_s']>=300])
    assert len(groups['ge120s'])==85 and len(groups['ge300s'])==25
    def stats(ids,arm):
        rows=[records[q,arm] for q in ids]
        return dict(questions=len(rows),correct=sum(r['correct'] for r in rows),
            content_accuracy=sum(r['correct'] for r in rows)/len(rows) if rows else None,
            parsed=sum(r['parsed_choice'] is not None for r in rows),format_ok=sum(r['format_ok'] for r in rows),
            QA_seconds=sum(r['QA_total_seconds'] for r in rows))
    scores={g:{a:stats(ids,a) for a in ARMS} for g,ids in groups.items()}
    bytype={cat:{a:stats([q for q,v in qs.items() if v['question_category']==cat],a) for a in ARMS}
        for cat in sorted({v['question_category'] for v in qs.values()})}
    comparisons={}
    for a,b in [(a,ARMS[0]) for a in ARMS[1:]]+[(ARMS[-1],ARMS[1]),(ARMS[-1],ARMS[2])]:
        delta={q:int(records[q,a]['correct'])-int(records[q,b]['correct']) for q in qs}
        comparisons[a+'_vs_'+b]={g:dict(net=sum(delta[q] for q in ids),
            corrected_ids=[q for q in ids if delta[q]>0],new_error_ids=[q for q in ids if delta[q]<0],
            both_wrong_ids=[q for q in ids if not records[q,a]['correct'] and not records[q,b]['correct']],
            reference_correct_questions=sum(records[q,b]['correct'] for q in ids),
            exploratory_source_interval=grouped_interval(ids,qs,delta)) for g,ids in groups.items()}
    costs={}
    for arm in ARMS:
        cs=[c for c in completed if c['task']['kind']=='cost' and c['task']['arms']==[arm]];assert len(cs)==4
        profiles=[p for c in cs for p in c['profiles']]
        assert len(profiles)==12 and len({p['video_id'] for p in profiles})==12 and sum(c['answers'] for c in cs)==57
        costs[arm]=dict(total_wall_seconds=sum(c['wall_seconds'] for c in cs),sources=12,questions=57,
            peak_GPU_allocated_bytes=max(c['CUDA_peak_allocated_bytes'] for c in cs),
            peak_GPU_reserved_bytes=max(c['CUDA_peak_reserved_bytes'] for c in cs),
            last_RSS_max_bytes=max(c['last_RSS_bytes'] for c in cs),
            persistent_bank_selector_max_bytes=max(p['state']['banks']['banks'][arm]['standalone_persistent_tensor_bytes'] for p in profiles),
            decode_seconds=sum(p['source']['decode_seconds'] for p in profiles),
            canvas_seconds=sum(p['source']['transform_seconds'] for p in profiles),
            preprocessing_seconds=sum(p['timing']['preprocess_seconds'] for p in profiles),
            vision_seconds=sum(p['timing']['vision_seconds'] for p in profiles),
            update_seconds=sum(p['timing']['bank_or_main_update_seconds'] for p in profiles),
            QA_seconds=sum(p['QA_seconds'] for p in profiles),audit_seconds=sum(p['audit_seconds'] for p in profiles),
            excluded_load_seconds=sum(c['load_seconds'] for c in cs),excluded_integrity_seconds=sum(c['integrity_seconds'] for c in cs),
            source_wall_seconds={p['video_id']:p['source_wall_seconds'] for p in profiles},
            independent_method_processes=True,balanced_order_four_disjoint_blocks=True,OS_cache_not_flushed=True)
    gates={}
    for a in ARMS[1:]:
        pair=comparisons[a+'_vs_'+ARMS[0]];ratio=costs[a]['total_wall_seconds']/costs[ARMS[0]]['total_wall_seconds']
        gates[a]=dict(all478_net=pair['all']['net'],old85_net=pair['ge120s']['net'],wall_ratio=ratio,
            operational_followup_gate=(pair['all']['net']>=-4 and pair['ge120s']['net']>=0 and ratio<=.95),
            not_statistical_noninferiority=True,default_enabled=False)
    result=dict(status='complete',scores=scores,by_type=bytype,comparisons=comparisons,
        independent_cost=costs,gates=gates,content_and_format_separate=True,
        end_gap_is_proxy_not_last_unique_evidence=True,reference_wrong_not_intrinsically_unanswerable=True,
        all478_opened_development=True,default_changed=False,not_edge_or_1x=True,
        no_automatic_R3_submission=True,all_failures_retained=True)
    save(OUT/'quality_cost_analysis.json',result);save(OUT/'scored_rows_LOCAL_ONLY.json',list(records.values()))
    print(dict(status='complete',scores=scores['all'],cost=costs,gates=gates),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);main(p.parse_args().results)
