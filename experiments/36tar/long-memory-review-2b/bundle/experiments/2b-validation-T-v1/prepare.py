"""Freeze unopened-source T candidates from local official annotations only.

Metadata proxies are not content admission, causal evidence, or model results.
No network, media decoding, model imports, or GPU submission in this module.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/2b_validation_T_prepare_20261020_01'
UPSTREAM = ROOT / 'experiments/2b-memory-eval/data/upstream/nextgqa'
SEED = 'T-independent-source-20261020-v1:'
OBSERVATION = {'TN', 'TP', 'TC', 'CH'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rank(value):
    return hashlib.sha256((SEED + str(value)).encode()).hexdigest()


def triplet(questions):
    duration = questions[0]['duration_s']
    early = [q for q in questions if q['task'] in OBSERVATION
             and q['evidence_gap_proxy_s'] >= 18 and q['evidence_span_proxy_s'] <= 12]
    recent = [q for q in questions if q['task'] in OBSERVATION
              and min(a for a, b in q['evidence_intervals']) >= duration - 16
              and q['evidence_gap_proxy_s'] <= 8]
    sequence = [q for q in questions if q['task'] in {'TN', 'TP', 'CW'}]
    choices = [x for x in itertools.product(early, recent, sequence)
               if len({q['id'] for q in x}) == 3]
    if not choices:
        return None
    return min(choices, key=lambda x: (
        x[2]['task'] == 'CW', -x[0]['evidence_gap_proxy_s'],
        x[0]['evidence_span_proxy_s'], x[1]['evidence_gap_proxy_s'],
        rank('|'.join(q['id'] for q in x))))


def main():
    OUT.mkdir(exist_ok=False)
    old_path = ROOT / 'runs/2b_collaboration_sync_20261020_01/media_manifest124.json'
    old = read(old_path)
    assert len(old) == 124
    excluded = {x['source_id'] for x in old if x['dataset'] == 'nextgqa'}
    assert len(excluded) == 84
    ground = read(UPSTREAM / 'gsub_val.json')
    mapping = read(UPSTREAM / 'map_vid_vidorID.json')
    groups = defaultdict(list)
    with (UPSTREAM / 'val.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            video = row['video_id']
            if video in excluded:
                continue
            g = ground.get(video, {})
            duration = g.get('duration', 0)
            intervals = g.get('location', {}).get(row['qid'], [])
            options = [row[f'a{i}'] for i in range(5)]
            if not duration or not intervals or options.count(row['answer']) != 1:
                continue
            if not all(0 <= a < b <= duration + .1 for a, b in intervals):
                continue
            groups[video].append(dict(
                id=f'nextgqa:val:{video}:{row["qid"]}', dataset='nextgqa',
                source_id=video, source_path=mapping[video], source_group_id='vidor:' + video,
                task=row['type'], question=row['question'], options=options,
                answer_index=options.index(row['answer']), evidence_intervals=intervals,
                duration_s=duration, query_time_s=duration, query_policy='original_full_clip_end',
                evidence_gap_proxy_s=duration - max(b for a, b in intervals),
                evidence_span_proxy_s=max(b for a, b in intervals) - min(a for a, b in intervals),
                split='T_unopened_source_candidate', media_review_status='not_acquired_not_viewed',
                formal_evaluation_eligible=False, empirical_difficulty='not_measured',
                old_evidence_necessity_proven=False))
    assert len(groups) == 483
    triples = {s: triplet(qs) for s, qs in groups.items()}
    pools = {
        '30-59': sorted((s for s in groups if 30 <= groups[s][0]['duration_s'] < 60
                         and triples[s]), key=rank),
        '60-119': sorted((s for s in groups if 60 <= groups[s][0]['duration_s'] < 120
                          and triples[s]), key=rank),
        '120+': sorted((s for s in groups if groups[s][0]['duration_s'] >= 120), key=rank)}
    assert len(pools['30-59']) >= 20 and len(pools['60-119']) >= 19
    assert len(pools['120+']) == 1
    selected = pools['30-59'][:20] + pools['60-119'][:19] + pools['120+']
    archive_path = ROOT / 'runs/nextqa_zip_preflight_20261016_01/directory/archive_index.json'
    archive = {x['path']: x for x in read(archive_path)['entries']}
    roster, sources = [], []
    for video in selected:
        if triples[video]:
            chosen = list(zip(triples[video], [
                'early_brief_observation_PROXY', 'recent_observation_PROXY',
                'order_or_visible_cause_PENDING_REVIEW']))
        else:
            chosen = [(q, 'long_duration_supplement_PENDING_REVIEW')
                      for q in sorted(groups[video], key=lambda q: rank(q['id']))]
            assert len(chosen) == 2 and groups[video][0]['duration_s'] == 180
        qs = [dict(q, selection_role=role) for q, role in chosen]
        roster.extend(qs)
        name = 'NExTVideo/' + mapping[video] + '.mp4'
        asset = archive[name]
        sources.append(dict(source_id=video, source_path=mapping[video],
            source_group_id='vidor:' + video, canonical_identity_only=True,
            filming_session_independence='pending_pixels_and_parent_review',
            official_archive_path=name, bytes=asset['bytes'], official_zip_crc32=asset['crc32'],
            duration_s=qs[0]['duration_s'], question_ids=[q['id'] for q in qs]))
    assert len(roster) == len({q['id'] for q in roster}) == 119
    assert len(sources) == len({s['source_id'] for s in sources}) == 40
    assert not {s['source_id'] for s in sources} & excluded
    assert sum(s['bytes'] for s in sources) <= 768 * 2**20
    inherited = ROOT / 'runs/2b_short_event_prepare_20261020_01/validation_T_admission_protocol.json'
    inputs = [old_path, UPSTREAM/'val.csv', UPSTREAM/'gsub_val.json',
              UPSTREAM/'map_vid_vidorID.json', archive_path, inherited]
    save(OUT/'candidate_questions119.json', roster)
    save(OUT/'acquisition_proposal.json', dict(
        status='frozen_metadata_candidates_not_acquired_or_qualified', sources=sources,
        total_media_bytes=sum(s['bytes'] for s in sources),
        repository='jinyoungkim/NExT-GQA', revision='5d1c262802efe7e5331629ca01e67916867590a5',
        origin='previously used third-party redistribution; not official hosting',
        integrity='official ZIP size/CRC consistency plus pinned mirror LFS SHA256',
        official_CRC_is_not_cryptographic_authentication=True,
        destination='datasets/2b-memory-eval/T-new40-20261020', media_byte_cap=768*2**20,
        wall_seconds=2700, maximum_attempts=1, automatic_retries=0,
        cluster_operations=0, GPU_jobs=0, authorization='not_yet_requested'))
    save(OUT/'review_template119.json', [dict(original=q, content_verdict='pending',
        witness_pts_s=[], late_repetition_or_context_cues='not_reviewed',
        visibility_under_fixed_input='pending', actual_recent8_coverage='pending',
        parent_identity='VIDOR_ID_only_pending_cross_split_review', reviewer='pending',
        independent_second_review='not_performed', admission_reason=None,
        formal_evaluation_eligible=False) for q in roster])
    summary = dict(status='metadata_roster_frozen_NOT_content_admitted',
        sources=40, original_questions=119, target_question_shortfall=1,
        shortfall_reason='only remaining >=120sec source has two original annotated questions; no fabricated third',
        remaining_official_sources=483, remaining_annotated_questions=sum(map(len, groups.values())),
        excluded_previously_acquired_sources=124, excluded_nextgqa_sources=84,
        feasible_triplet_sources={k: len(v) for k, v in pools.items()},
        duration_strata={'30-59':20, '60-119':19, '120+':1},
        min_duration_s=min(s['duration_s'] for s in sources),
        max_duration_s=max(s['duration_s'] for s in sources),
        roles=dict(Counter(q['selection_role'] for q in roster)),
        official_tasks=dict(Counter(q['task'] for q in roster)),
        bytes=sum(s['bytes'] for s in sources), source_sha256=sha(__file__),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        output_sha256={n:sha(OUT/n) for n in [
            'candidate_questions119.json','acquisition_proposal.json','review_template119.json']},
        seed=SEED, selection='hash-ranked sources within fixed duration strata; exact triplet rule in locked source',
        recent_proxy='all official evidence starts within final16s and ends within final8s',
        early_proxy='observable task, all evidence ends >=18s before query, total interval span <=12s',
        third_proxy='distinct TN/TP original order question preferred over CW cause question; visible cause requires pixels',
        inherited_protocol_sha256=sha(inherited), no_model_conditioned_selection=True,
        models_loaded=0, media_downloaded=0, content_reviews_completed=0,
        formal_admitted_questions=0, known_parent_overlap='requires actual pixel review; IDs alone not proof',
        stop='one qualification pass, keep ambiguities; no replacement based on model performance',
        long_duration_coverage_gap='only one180s source; no adequate multi-minute population or streaming-necessity claim')
    save(OUT/'metadata_freeze.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
