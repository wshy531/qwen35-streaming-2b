"""Register published diagnostic anchors, keeping memory claims separate."""
from collections import Counter
import hashlib

from package_dataset import canonical, DEST, ROOT, read, save
from audit_validation import short_provenance

REVIEW = ROOT / 'runs/diagnosis_expansion_review_20261105_01'


def main():
    receipt = read(REVIEW / 'preparation_receipt.json')
    assert receipt['status'] == 'complete' and receipt['technical_pass'] == 30
    packets = read(REVIEW / 'review_packets.json')
    assert len(packets) == len({r['source_id'] for r in packets}) == 30
    official = read(REVIEW / 'official_anchor_labels_LOCAL_ONLY.json')
    labels = {k: ord(v.strip()[0]) - 65 for k, v in official.items()}
    old_media = read(ROOT / 'runs/streamingbench_official_tail_20261031_01/complete_manifest.json')['files']
    assert {r['media_sha256'] for r in packets}.isdisjoint(r['sha256'] for r in old_media)
    assert len({r['media_sha256'] for r in packets}) == 30
    _, _, _, known_hashes = short_provenance()
    assert {r['media_sha256'] for r in packets}.isdisjoint(known_hashes)
    bindings, rows = {}, []
    for packet in packets:
        for slot in ('early', 'current'):
            original = packet[slot + '_question']
            source = dict(original, media_path=packet['media_path'], media_sha256=packet['media_sha256'],
                          duration_s=packet['duration_s'])
            row = canonical(source, 'StreamingBench', 'published_anchor_check', labels, bindings)
            row.update(anchor_role=slot, memory_requirement='not_established',
                       capture_isolation='unverified_original_capture',
                       review_packet='runs/diagnosis_expansion_review_20261105_01/' + packet['source_id'][:-4],
                       annotation_reference=dict(row=original['annotation_row'], index=original['question_index']),
                       eligible_for_current_history_split=False)
            status = packet['early_sampler' if slot == 'early' else 'late_sampler']
            row['effective_committed_pts_s'] = status['last_committed_pts']
            row['query_to_last_committed_gap_s'] = row['query_time_s'] - status['last_committed_pts']
            assert row['query_to_last_committed_gap_s'] >= -1e-8
            rows.append(row)
    assert len(rows) == len({r['id'] for r in rows}) == 60
    registry = read(DEST / 'registry.json')
    old_hashes = {k: v['sha256'] for k, v in registry['suites'].items()}
    assert 'diagnosis_anchors' not in registry['suites']
    path = DEST / 'diagnosis_anchors.json'
    save(path, rows)
    registry['suites']['diagnosis_anchors'] = dict(path=path.name, questions=60, source_files=30,
        dataset_counts={'StreamingBench': 60}, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        runnable=True, score_groups_separately=False,
        purpose='published-question readout check; not a certified current/history dependency panel')
    keys = read(DEST / 'answer_keys_LOCAL_ONLY.json')
    assert set(keys).isdisjoint(labels)
    keys.update(labels)
    registry['unique_questions'] = len(keys)
    save(DEST / 'answer_keys_LOCAL_ONLY.json', keys)
    save(DEST / 'registry.json', registry)
    for key, value in old_hashes.items():
        assert hashlib.sha256((DEST / registry['suites'][key]['path']).read_bytes()).hexdigest() == value
    stats = dict(sources=30, official_questions=60,
                 types=dict(Counter(q['type'] for q in rows)),
                 early_query_range=[min(r['early_query_s'] for r in packets), max(r['early_query_s'] for r in packets)],
                 late_query_range=[min(r['late_query_s'] for r in packets), max(r['late_query_s'] for r in packets)],
                 media_duration_range=[min(r['duration_s'] for r in packets), max(r['duration_s'] for r in packets)],
                 explicit_gap_range=[min(r['late_query_s']-r['early_query_s'] for r in packets),
                                     max(r['late_query_s']-r['early_query_s'] for r in packets)],
                 not_independent_60_sources=True, original_capture_isolation_verified=False,
                 current_history_requirement_verified=False,
                 known_project_media_sha_checked=len(known_hashes),
                 committed_cutoff_lag_range_s=[min(q['query_to_last_committed_gap_s'] for q in rows),
                                              max(q['query_to_last_committed_gap_s'] for q in rows)],
                 old_suite_hashes_preserved=True)
    save(REVIEW / 'release_summary.json', stats)
    print(stats)


if __name__ == '__main__':
    main()
