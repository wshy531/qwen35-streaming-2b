"""Post-run descriptive audit; does not alter the frozen primary scoring."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

from prepare import OUT, read, save, sha


def audit(result):
    rows = read(OUT/'scorer_rows_LOCAL_ONLY.json')
    requests = {r['id']: r for r in read(OUT/'requests.json')}
    protocol = OUT/'runtime_inputs/runtime_protocol.json'
    completions = {s: read(result/s/'completed.json') for s in ['native','memory']}
    loaded = {s: read(result/s/'loaded.json') for s in completions}
    assert all(c['protocol_sha256'] == sha(protocol) for c in completions.values())
    assert sum(c['model_loads'] for c in completions.values()) == 2
    assert sum(c['generated_tokens'] for c in completions.values()) <= 6272
    scores = read(result/'per_question_scores.json')
    by = {(p['id'], p['arm']): p for p in scores}
    arms = list(dict.fromkeys(p['arm'] for p in scores))
    for p in scores:
        expected = hashlib.sha256(json.dumps(requests[p['id']]['model_visible']['frames'],
                                            sort_keys=True).encode()).hexdigest()
        assert p['input_frames_sha256'] == expected
        assert len(p['generated_ids']) <= 8
    timeline = read(result/'memory/timelines.json')
    assert len(timeline) == 39
    storage = defaultdict(list)
    for t in timeline:
        for arm, s in t['states'].items():
            assert s['tokens'] <= 384
            nodes = s.get('nodes', [])
            f = s.get('forest', s)
            tensor_bytes = s.get('candidate_tensor_bytes',
                                s.get('resident_tensor_bytes', s.get('tensor_bytes')))
            storage[arm].append(dict(media=t['media'], tokens=s['tokens'],
                tensor_bytes=tensor_bytes, detector_bytes=t['states']['M2'].get('detector_state_bytes',0) if arm in ['M1','M2','B02_cap4'] else 0,
                forced_seals=f.get('forced_seals',0), within_merges=f.get('within_merges',0),
                cross_merges=f.get('long_merges',0),
                max_depth=max((n['depth'] for n in nodes), default=0),
                earliest_begin=min((n['begin'] for n in nodes), default=None)))
    summary = {}
    for arm, states in storage.items():
        summary[arm] = dict(sources=len(states), maximum_visual_tokens=max(s['tokens'] for s in states),
            maximum_persistent_feature_tensor_bytes=max(s['tensor_bytes'] for s in states),
            maximum_shared_event_detector_bytes=max(s['detector_bytes'] for s in states),
            forced_seals=sum(s['forced_seals'] for s in states),
            forced_seal_sources=sum(s['forced_seals']>0 for s in states),
            within_merges=sum(s['within_merges'] for s in states),
            cross_merges=sum(s['cross_merges'] for s in states),
            maximum_final_node_depth=max(s['max_depth'] for s in states))
    contrasts = {}
    for a,b in [('M0','M2'),('M2','B02_cap4'),('M0','reservoir4')]:
        changes = []
        for r in rows:
            left, right = by[r['id'],a], by[r['id'],b]
            if left['content_correct'] == right['content_correct']:
                continue
            changes.append(dict(id=r['id'],source=r['source_path'],stratum=r['stratum'],
                role=r['selection_role'],memory_character=r['memory_character'],
                direction='correction' if right['content_correct'] else 'new_loss',
                question=r['original']['question'],options=r['original']['options'],
                official_answer_index=r['original']['answer_index'],
                reference_text=left['text'],candidate_text=right['text'],
                review_note=r['note'],late_cues=r['late_repetition_or_context_cues']))
        contrasts[b+'_vs_'+a] = changes
    speed = {}
    for a in arms:
        records = [p for p in scores if p['arm']==a]
        speed[a] = dict(answers=len(records),
            sum_answer_seconds=sum(p['seconds'] for p in records),
            median_answer_seconds=statistics.median(p['seconds'] for p in records),
            maximum_visual_read_tokens=max(p.get('visual_tokens',0) for p in records),
            generated_tokens=sum(len(p['generated_ids']) for p in records))
    long_rows = [r for r in rows if r['original']['duration_s'] >= 120]
    save(result/'descriptive_audit_v2.json',dict(status='completed',
        scope='post-run descriptive audit; primary frozen analysis unchanged',
        protocol_sha256=sha(protocol), completions=completions, loaded=loaded,
        storage_summary=summary,storage_per_source=dict(storage),
        answer_timing_descriptive_only=speed,paired_changes=contrasts,
        long_supplement=[dict(id=r['id'],stratum=r['stratum'],
            scores={a:by[r['id'],a]['content_correct'] for a in arms}) for r in long_rows],
        limitations=['shared detector bytes apply to event-based policies; zero absent original fields did not prove no detector',
          'final bank snapshots are not full trajectory peak measurements',
          'persistent feature bytes exclude model, reader caches, temporary exports and full-reference cache',
          'RSS checked below20GiB after each answer; exact process RSS peak was not recorded',
          'fixed arm order and one pass cannot establish speed advantage',
          'follow-up error interpretation is descriptive, not a new hypothesis test']))
    print(json.dumps(dict(storage=summary,timing=speed,long_supplement=len(long_rows))))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
