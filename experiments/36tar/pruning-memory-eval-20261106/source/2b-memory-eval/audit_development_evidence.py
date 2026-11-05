"""Read-only upstream metadata audit; no media download or model answers.

Only development questions are analyzed for task suitability. All source paths
are checked for split leakage. Filename identities remain provisional until
provenance review; unique filenames are not proof of independent recordings.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from audit_candidates import validate
from audit_sampling import query_snapshot


def normalize(text):
    return ' '.join(text.lower().strip(' .?').split())


def no_evidence_option_rule(row):
    """Dataset shortcut diagnostic, never used as a model answer policy."""
    matches = [i for i,text in enumerate(row['options']) if normalize(text)=='unable to answer']
    return matches[0] if len(matches)==1 else None


def source_identity(dataset, source):
    stem = PurePosixPath(source).stem
    family = source.split('/')[0]
    if dataset == 'nextgqa':
        return 'nextqa-asset:'+stem, 'inferred_author_asset_filename'
    if family in {'cross_task','COIN','youcook2'} and re.fullmatch(r'[A-Za-z0-9_-]{11}',stem):
        return 'youtube:'+stem, 'inferred_author_asset_filename'
    if family == 'hirest':
        match = re.fullmatch(r'([A-Za-z0-9_-]{11})_\d+_\d+',stem)
        if match:
            return 'youtube:'+match[1], 'inferred_parent_from_author_clip_filename'
    # Never pretend that an Ego4D clip UUID resolves its parent video UUID.
    return dataset+':asset:'+source, 'parent_recording_unresolved'


def leakage_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        key,status = source_identity(row['dataset'],row['source_path'])
        groups[key].append(dict(id=row['id'],split=row['split'],source=row['source_path'],
                                identity_status=status))
    return {key:items for key,items in groups.items()
            if len({r['split'] for r in items}) > 1}


def parse_crosstask(path):
    with zipfile.ZipFile(path) as archive:
        text = archive.read('crosstask_release/tasks_primary.txt').decode('utf-8')
        tasks = {block.splitlines()[0]:block.splitlines()[4].split(',')
                 for block in text.strip().split('\n\n')}
        by_video = {}
        for name in archive.namelist():
            match = re.fullmatch(r'crosstask_release/annotations/(\d+)_(.+)\.csv',name)
            if not match or match[1] not in tasks:
                continue
            # Metadata only; read each small member, never extract archive paths.
            steps = [dict(step=int(n),action=normalize(tasks[match[1]][int(n)-1]),
                          start_s=float(a),end_s=float(b)) for n,a,b in
                     csv.reader(io.StringIO(archive.read(name).decode('utf-8')))]
            by_video[match[2]] = dict(member=name,steps=steps)
    return by_video


def crosstask_question(row, annotations):
    entry = annotations.get(PurePosixPath(row['source_path']).stem)
    match = re.fullmatch(r'what does the person do (before|after) (.+)',normalize(row['question']))
    if not entry or not match:
        return dict(status='source_annotation_or_question_pattern_unavailable')
    answer = normalize(row['options'][row['answer_index']])
    anchor = [s for s in entry['steps'] if s['action']==match[2]]
    evidence = [s for s in entry['steps'] if s['action']==answer]
    query = row['query_time_s']
    past = [s for s in evidence if s['end_s'] <= query+1e-9]
    ongoing = [s for s in evidence if s['start_s'] <= query+1e-9 < s['end_s']]
    future = [s for s in evidence if s['start_s'] > query+1e-9]
    if not anchor or not evidence:
        status = 'source_action_name_unmatched'
    elif future and not past and not ongoing:
        status = 'future_answer_evidence_not_a_past_recall_probe'
    elif past and not future and not ongoing:
        status = 'source_supports_past_answer_visual_review_still_required'
    elif ongoing:
        status = 'answer_action_ongoing_at_query_requires_task_review'
    else:
        status = 'repeated_answer_action_requires_occurrence_disambiguation'
    return dict(status=status,source_member=entry['member'],relation=match[1],
        anchor_action=match[2],anchor_intervals=[[s['start_s'],s['end_s']] for s in anchor],
        answer_action=answer,answer_intervals=[[s['start_s'],s['end_s']] for s in evidence],
        anchor_start_matches_query=any(abs(s['start_s']-query)<1e-6 for s in anchor),
        earliest_answer_start_minus_query_s=min((s['start_s']-query for s in evidence),default=None),
        evidence_scope='author_intervals_only_not_media_inspection_or_official_benchmark_erratum')


def audit(rows, crosstask, coin=None):
    validate(rows)
    records = []
    for row in rows:
        if row['split'] != 'development':
            continue
        item = dict(id=row['id'],dataset=row['dataset'],source_path=row['source_path'],
            task=row['task'],query_time_s=row['query_time_s'],
            official_record_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,
                ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
        if row['source_path'].startswith('cross_task/'):
            item['source_evidence_audit'] = crosstask_question(row,crosstask)
        elif row['source_path'].startswith('COIN/'):
            item['source_evidence_audit'] = crosstask_question(row,coin or {})
        elif row['dataset']=='nextgqa':
            intervals = row['evidence_intervals']
            times = [i*2. for i in range(int(row['query_time_s']//2)+1)]
            frames = [dict(frame_id=i,pts_s=t) for i,t in enumerate(times)]
            snapshot = query_snapshot(frames,row['query_time_s'],intervals)
            snapshot['grid_any_annotated_interval_hit'] = any(
                a <= t <= b for a,b in intervals for t in times)
            item['ideal_grid_audit'] = dict(
                label='metadata_only_ideal_grid_not_decoded_frames_or_visual_readability',
                interval_semantics='author_corresponding_segments_not_independently_verified_as_all_necessary',
                intervals=intervals,gap_since_last_end_s=row['evidence_gap_s'],
                full_all_spans_hit=snapshot['full_history_all_verified_intervals_hit'],
                full_any_span_hit=snapshot['grid_any_annotated_interval_hit'],
                M0_all_spans_hit=snapshot['M0']['all_verified_intervals_hit'],
                M1_all_spans_hit=snapshot['M1']['all_verified_intervals_hit'],
                M0_pair_indices=snapshot['M0']['pair_indices'],
                M1_pair_indices=snapshot['M1']['pair_indices'])
        records.append(item)
    sources = {(r['dataset'],r['source_path']) for r in rows}
    unresolved = [dict(dataset=d,source_path=s,identity=source_identity(d,s)[0])
                  for d,s in sorted(sources) if source_identity(d,s)[1]=='parent_recording_unresolved']
    gqa = [r['ideal_grid_audit'] for r in records if 'ideal_grid_audit' in r]
    development = [r for r in rows if r['split']=='development']
    rule = [(r,no_evidence_option_rule(r)) for r in development]
    triggered = [(r,i) for r,i in rule if i is not None]
    return dict(status='development_metadata_audit_not_frozen_or_model_evaluation',
        development_questions=len(records),
        action_source_statuses=dict(Counter(r['source_evidence_audit']['status'] for r in records
                                       if 'source_evidence_audit' in r)),
        nextgqa_ideal_grid=dict(questions=len(gqa),
            full_any=sum(r['full_any_span_hit'] for r in gqa),
            full_all=sum(r['full_all_spans_hit'] for r in gqa),
            M0_all=sum(r['M0_all_spans_hit'] for r in gqa),
            M1_all=sum(r['M1_all_spans_hit'] for r in gqa)),
        cross_split_filename_identity_conflicts=leakage_groups(rows),
        unresolved_parent_sources=unresolved,questions=records,
        option_cue_audit=dict(rule='select_unique_unable_to_answer_option_without_seeing_video',
            triggered_questions=len(triggered),
            triggered_ids=[r['id'] for r,i in triggered],
            matched_original_labels=sum(i==r['answer_index'] for r,i in triggered),
            HLD_questions=sum(r['task']=='HLD' for r in development),
            non_HLD_with_unable_option=sum(r['task']!='HLD' for r,i in triggered),
            interpretation='dataset_format_shortcut_diagnostic_not_model_accuracy_or_proof_of_grounded_refusal'),
        limitations=['no_new_media_or_model_answers',
            'filename_identity_check_is_not_full_parent_or_perceptual_dedup',
            'ideal_grid_hit_is_not_actual_sampling_or_answer_accuracy',
            'all_annotated_spans_are_not_proven_jointly_necessary',
            'do_not_silently_modify_official_query_or_ground_truth'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--crosstask-zip',required=True,type=Path)
    p.add_argument('--coin-selected-json',type=Path)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    rows = json.loads(args.manifest.read_text(encoding='utf-8'))
    coin = {}
    if args.coin_selected_json:
        for key,value in json.loads(args.coin_selected_json.read_text(encoding='utf-8')).items():
            coin[key] = dict(member='COIN.json/database/'+key,steps=[dict(
                action=normalize(s['label']),start_s=s['segment'][0],end_s=s['segment'][1])
                for s in value['annotation']])
    result = audit(rows,parse_crosstask(args.crosstask_zip),coin)
    result['inputs_sha256'] = {str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in (args.manifest,args.crosstask_zip)}
    if args.coin_selected_json:
        result['inputs_sha256'][str(args.coin_selected_json)] = hashlib.sha256(
            args.coin_selected_json.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in
                     {'questions','unresolved_parent_sources','inputs_sha256'}},ensure_ascii=False))


if __name__ == '__main__':
    main()
