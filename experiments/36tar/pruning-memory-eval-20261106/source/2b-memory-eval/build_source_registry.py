"""Trace canonical recording/scene identity separately from file identity.

Known publisher identifiers are not proof of perceptual independence. Unresolved
Ego4D clip parents remain unresolved and are blocked from a verified-source core.
"""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path,PurePosixPath
import re


def resolve(dataset,source,vidor,episodes,star,hirest=None):
    stem=PurePosixPath(source).stem
    family=source.split('/')[0]
    if dataset=='nextgqa':
        if vidor.get(stem)!=source:raise ValueError('author VIDOR mapping mismatch')
        return dict(source_group_id='vidor:'+stem,parent_identity_status='verified',
            basis='official_nextgqa_map_vid_vidorID',identity_level='author_canonical_video_id',
            content_domain='naturalistic_recorded_video',evidence_file='map_vid_vidorID.json')
    if family in ('cross_task','COIN','youcook2') and re.fullmatch(r'[A-Za-z0-9_-]{11}',stem):
        return dict(source_group_id='youtube:'+stem,parent_identity_status='verified',
            basis='original_upload_identifier_in_official_OVO_asset_path',
            identity_level='youtube_upload_not_actor_or_filming_session',
            content_domain='recorded_instructional_video',evidence_file='pinned_OVO_TAR_index')
    if family=='hirest':
        match=re.fullmatch(r'([A-Za-z0-9_-]{11})_(\d+)_(\d+)',stem)
        if match:
            bounds=[int(match[2]),int(match[3])]
            matches=(hirest or {}).get(match[1],[])
            if hirest is not None and bounds not in matches:
                raise ValueError('HiREST author trim bounds mismatch')
            return dict(source_group_id='youtube:'+match[1],parent_identity_status='verified',
                basis='OVO_filename_plus_HiREST_author_bounds' if hirest is not None else
                    'original_upload_and_trim_bounds_in_official_OVO_filename',
                identity_level='youtube_upload',parent_trim_s=bounds,
                content_domain='recorded_web_video',evidence_file='source_metadata/hirest_all_data_val.json'
                    if hirest is not None else 'pinned_OVO_TAR_index')
    if family=='OpenEQA':
        episode='hm3d-v0/'+stem
        if episode not in episodes:raise ValueError('OpenEQA author episode absent')
        scene=stem.split('-hm3d-')[1]
        return dict(source_group_id='hm3d-scene:'+scene,parent_identity_status='verified',
            basis='official_open_eqa_episode_history',identity_level='scan_scene_conservative_group',
            episode_history=episode,content_domain='simulated_tour_of_real_scanned_scene',
            evidence_file='source_metadata/open-eqa-v0.json')
    if family=='star':
        if stem not in star:raise ValueError('STAR author video_id absent')
        return dict(source_group_id='charades:'+stem,parent_identity_status='verified',
            basis='STAR_author_Video_Segments_video_id_with_Charades_provenance',
            identity_level='charades_canonical_recording_id',
            author_segment_count=star[stem],content_domain='scripted_indoor_human_video',
            evidence_file='source_metadata/STAR_Video_Segments.csv')
    return dict(source_group_id=None,parent_identity_status='unresolved',
        basis='clip_UUID_does_not_identify_parent_video',identity_level='asset_only',
        content_domain='egocentric_recorded_video',evidence_file=None)


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    upstream=root/'experiments/2b-memory-eval/data/upstream'
    candidate=root/'experiments/2b-memory-eval/data/candidate_manifest.json'
    rows=json.loads(candidate.read_text(encoding='utf-8'))
    vidor=json.loads((upstream/'nextgqa/map_vid_vidorID.json').read_text(encoding='utf-8'))
    episodes={q['episode_history'] for q in json.loads((run/'source_metadata/open-eqa-v0.json').read_text(encoding='utf-8'))}
    star=Counter(r['video_id'] for r in csv.DictReader((run/'source_metadata/STAR_Video_Segments.csv').open(encoding='utf-8-sig',newline='')))
    hirest=defaultdict(list)
    # Identity/bounds only, including holdout; no holdout step labels used here.
    for videos in json.loads((run/'source_metadata/hirest_all_data_val.json').read_text(encoding='utf-8')).values():
        for video,entry in videos.items():
            if entry.get('bounds'):hirest[PurePosixPath(video).stem].append(entry['bounds'])
    result=[]
    for dataset,source in dict.fromkeys((r['dataset'],r['source_path']) for r in rows):
        qs=[r for r in rows if r['dataset']==dataset and r['source_path']==source]
        identity=resolve(dataset,source,vidor,episodes,star,hirest)
        result.append(dict(dataset=dataset,source_path=source,split=qs[0]['split'],
            question_ids=[q['id'] for q in qs],**identity))
    groups=defaultdict(list)
    for item in result:
        if item['source_group_id']:groups[item['source_group_id']].append(item)
    conflicts={k:[dict(source_path=v['source_path'],split=v['split']) for v in values]
        for k,values in groups.items() if len({v['split'] for v in values})>1}
    report=dict(status='source_identifier_registry_not_full_perceptual_dedup_or_evaluation_freeze',
        candidate_manifest_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        sources=result,known_identifier_sources=sum(v['parent_identity_status']=='verified' for v in result),
        unresolved_sources=sum(v['parent_identity_status']=='unresolved' for v in result),
        cross_split_known_group_conflicts=conflicts,
        content_domains=dict(Counter(v['content_domain'] for v in result)),
        limits=['Verified means stated canonical identifier/scene grouping, not independent human subjects or complete overlap detection.',
            'Ego4D official anonymous metadata returned 403; no credentials inspected or access bypass attempted.',
            'Original split unchanged; unresolved sources retained in audit but cannot enter verified-source core.'])
    (run/'source_registry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (run/'来源分组审核.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['dataset','source_path','split','source_group_id','parent_identity_status','identity_level','content_domain','basis']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','limits')},ensure_ascii=False))


if __name__=='__main__':main()
