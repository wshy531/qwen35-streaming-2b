"""Frozen content-first scoring and an independent LS cost frontier; labels stay local."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
from resolution_common import ROOT,OUT,SPECS,ARMS,POLICIES,read,save,sha,verify
from sember_protocol import parse_choice,format_ok
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_sember_main import grouped_interval


def lines(p):
    import json
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]


def main(results):
    protocol=verify();assert read(results/'completed.json')['status']=='complete'
    labelpath=ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'
    assert sha(labelpath)==read(OUT/'local_analysis_identity.json')['labels_SHA']
    qs={q['question_id']:q for q in read(OUT/'inputs/requests.json')};labels={r['question_id']:r for r in read(labelpath)}
    records={};complete=[]
    for task in read(OUT/'inputs/tasks.json'):
        base=results/task['name'];completed=read(base/'completed.json');assert completed['status']=='complete';complete.append(completed)
        rows=lines(base/'answers.jsonl');events=defaultdict(list)
        for e in lines(base/'tokens.jsonl'):events[e['question_id'],e['arm']].append(e['token_id'])
        ids={q for q,r in qs.items() if task['videos'] is None or r['video_id'] in task['videos']}
        assert len(rows)==len(ids)*len(task['arms'])==completed['answers']
        assert len({(r['question_id'],r['arm']) for r in rows})==len(rows)
        for r in rows:
            assert r['generated_ids']==events[r['question_id'],r['arm']] and 1<=len(r['generated_ids'])<=16
            assert r['read_state_unchanged'] and r['branch_released']
            if task['kind']=='quality':records[r['question_id'],r['arm']]=r
            else:
                old=records[r['question_id'],r['arm']]
                for key in ('generated_ids','text','question_suffix_sha256','bounded_context_ids_sha256','visual_tokens'):assert r[key]==old[key]
    assert set(records)=={(q,a) for q in qs for a in ARMS}
    assert sum(c['answers']+c['warmup_answers'] for c in complete)==protocol['max_answers']
    assert sum(c['vision_encodings']+c['warmup_encodings'] for c in complete)==protocol['max_vision_encodings']
    for (qid,a),r in records.items():
        choice=parse_choice(r['text'],len(qs[qid]['options']))
        r.update(parsed_choice=choice,correct=choice==labels[qid]['correct_letter'],format_ok=format_ok(r['text'],len(qs[qid]['options'])))
    groups=dict(all=list(qs),ge120s=[q for q in qs if labels[q]['end_gap_proxy_s']>=120],
        ge300s=[q for q in qs if labels[q]['end_gap_proxy_s']>=300])
    assert len(groups['ge120s'])==85 and len(groups['ge300s'])==25
    def stats(ids,arm):
        rs=[records[q,arm] for q in ids]
        return dict(questions=len(rs),correct=sum(r['correct'] for r in rs),parsed=sum(r['parsed_choice'] is not None for r in rs),
            format_ok=sum(r['format_ok'] for r in rs),QA_seconds=sum(r['QA_total_seconds'] for r in rs))
    scores={g:{a:stats(ids,a) for a in ARMS} for g,ids in groups.items()}
    comparisons={}
    pairs=[(p+'_'+a,p+'_'+b) for p in POLICIES for a,b in
        [('highchange108','mid108'),('high192','mid108'),('highchange108','high192')]]
    pairs.append(('M0_highchange108','M0_highuniform108'))
    for a,b in pairs:
        delta={q:int(records[q,a]['correct'])-int(records[q,b]['correct']) for q in qs}
        comparisons[a+'_vs_'+b]={g:dict(net=sum(delta[q] for q in ids),
            corrected_ids=[q for q in ids if delta[q]>0],new_error_ids=[q for q in ids if delta[q]<0],
            exploratory_source_interval=grouped_interval(ids,qs,delta)) for g,ids in groups.items()}
    costs={}
    for arm in protocol['cost_views']:
        cs=[c for c in complete if c['task']['kind']=='cost' and c['task']['arms']==[arm]]
        assert len(cs)==3
        costs[arm]=dict(sources=sum(len(c['profiles']) for c in cs),questions=sum(c['answers'] for c in cs),
            total_wall_seconds=sum(c['wall_seconds'] for c in cs),
            peak_GPU_allocated_bytes=max(c['CUDA_peak_allocated_bytes'] for c in cs),
            peak_GPU_reserved_bytes=max(c['CUDA_peak_reserved_bytes'] for c in cs),
            max_last_RSS_bytes=max(c['last_RSS_bytes'] for c in cs),
            decode_seconds=sum(p['source']['decode_seconds'] for c in cs for p in c['profiles']),
            transform_seconds=sum(p['source']['transform_seconds'] for c in cs for p in c['profiles']),
            preprocessing_seconds=sum(p['timing']['preprocess_seconds'] for c in cs for p in c['profiles']),
            vision_seconds=sum(p['timing']['vision_seconds'] for c in cs for p in c['profiles']),
            update_seconds=sum(p['timing']['bank_or_main_update_seconds'] for c in cs for p in c['profiles']),
            QA_seconds=sum(p['QA_seconds'] for c in cs for p in c['profiles']),
            audit_seconds=sum(p['audit_seconds'] for c in cs for p in c['profiles']),
            max_persistent_bank_and_selector_bytes=max(p['state']['banks']['banks'][arm]['standalone_persistent_tensor_bytes'] for c in cs for p in c['profiles']),
            load_seconds=sum(read(results/c['task']['name']/'startup.json')['load_seconds'] for c in cs),integrity_seconds=sum(read(results/c['task']['name']/'startup.json')['integrity_seconds'] for c in cs),
            disjoint_source_blocks_not_three_repeats=True)
    base=costs['LS_mid108'];gate={}
    for a in ('LS_highchange108','LS_high192'):
        wall=costs[a]['total_wall_seconds']/base['total_wall_seconds'];mem=costs[a]['peak_GPU_allocated_bytes']/base['peak_GPU_allocated_bytes']
        net=comparisons[a+'_vs_LS_mid108']
        gate[a]=dict(wall_ratio=wall,peak_GPU_ratio=mem,
            candidate_for_followup_only=(all(net[g]['net']>=0 for g in ('all','ge120s')) and
                any(net[g]['net']>0 for g in ('all','ge120s')) and wall<=1.20 and mem<=1.25),
            default_enabled=False,operational_gate_not_significance_or_universal_value=True)
    bytype={cat:{a:stats([q for q,v in qs.items() if v['question_category']==cat],a) for a in ARMS}
        for cat in sorted({v['question_category'] for v in qs.values()})}
    result=dict(status='complete',all478_preserved=True,scores=scores,by_type=bytype,comparisons=comparisons,
        independent_LS_cost=costs,LS_followup_gates=gate,cost_not_generalized_to_other_policies=True,
        content_and_format_separate=True,default_changed=False,not_edge_or_1x_playback=True,
        all_outcomes_reported_even_if_gate_fails=True)
    save(OUT/'resolution_analysis.json',result);save(OUT/'scored_rows_LOCAL_ONLY.json',list(records.values()))
    print(dict(status='complete',scores=scores['all'],costs=costs,gates=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);main(p.parse_args().results)
