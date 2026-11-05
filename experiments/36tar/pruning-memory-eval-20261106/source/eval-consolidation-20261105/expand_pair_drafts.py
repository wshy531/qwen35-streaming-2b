"""Add visually supported paired drafts without relabelling the official anchors."""
import hashlib
import re

from package_dataset import canonical, DEST, ROOT, read, save

REVIEW = ROOT / 'runs/diagnosis_expansion_review_20261105_01'
# Fixed from visual inspection, without any model answers or algorithm scores.
# These are draft diagnostic pairs; independence and evidence-necessity controls remain separate.
SELECTED = ('sample_67_real.mp4', 'sample_59_real.mp4', 'sample_308_real.mp4',
            'sample_140_real.mp4', 'sample_157_real.mp4', 'sample_52_real.mp4',
            'sample_17_real.mp4', 'sample_152_real.mp4')


def main():
    packets = {r['source_id']: r for r in read(REVIEW / 'review_packets.json')}
    labels = read(DEST / 'answer_keys_LOCAL_ONLY.json')
    rows = read(DEST / 'diagnosis_candidate.json')
    assert len(rows) == 6
    notes = {r['source_id']: r for r in read(REVIEW / 'content_review_notes_LOCAL_ONLY.json')}
    for name in SELECTED:
        packet = packets[name]
        assert packet['technical_pass'] and notes[name]['category'] == 'provisional_pair'
        for role in ('history', 'current'):
            slot = 'early' if role == 'history' else 'current'
            original = packet[slot + '_question']
            new_id = 'sb-draft-' + name[:-4] + '-' + role
            source = dict(original, id=new_id, query_time_s=packet['late_query_s'],
                          media_path=packet['media_path'], media_sha256=packet['media_sha256'],
                          duration_s=packet['duration_s'], type=role,
                          length_bin='prefix<180s' if packet['late_query_s'] < 180 else 'prefix180-600s')
            if role == 'history':
                question = re.sub(r'\bright now\b', 'at that earlier time', original['question'], flags=re.I)
                question = re.sub(r'\bjust now\b', 'immediately before that earlier time', question, flags=re.I)
                source['question'] = f'Earlier, around {packet["early_query_s"]:g} seconds into the video: {question}'
            labels[new_id] = labels[original['id']]
            item = canonical(source, 'StreamingBench', 'single_assistant_diagnostic_draft', labels, {})
            item.update(diagnostic_role=role, original_task_type=original['type'],
                        pair_id=name[:-4], paired_anchor_id=original['id'],
                        evidence_bounds_s=[packet[slot + '_window']['timestamps'][0],
                                           packet[slot + '_window']['timestamps'][-1]] if role == 'history'
                            else [packet['recent_window']['timestamps'][0], packet['recent_window']['timestamps'][-1]],
                        recent_input_pts=packet['recent_window']['timestamps'],
                        reviewer='single_assistant_visual_review', independent_review=False,
                        history_dependency_controls='not_run',
                        content_review_reference='runs/diagnosis_expansion_review_20261105_01/content_review_notes_LOCAL_ONLY.json')
            rows.append(item)
    assert len(rows) == 22 and len({r['id'] for r in rows}) == 22
    save(DEST / 'diagnosis_candidate.json', rows)
    registry = read(DEST / 'registry.json')
    registry['suites']['diagnosis_candidate'].update(questions=len(rows), source_files=11,
        dataset_counts={'StreamingBench': len(rows)},
        sha256=hashlib.sha256((DEST / 'diagnosis_candidate.json').read_bytes()).hexdigest(),
        runnable=False, purpose='paired evidence drafts; not yet a certified dependency benchmark')
    registry['unique_questions'] = len(labels)
    save(DEST / 'registry.json', registry)
    save(DEST / 'answer_keys_LOCAL_ONLY.json', labels)
    save(REVIEW / 'paired_draft_release.json', dict(pairs=11, questions=22,
         new_pairs=8, original_three_pairs_preserved=True, model_outputs_used=False,
         formal_dependency_admissions=0, independent_review=False,
         next_step='Frozen recent-only / early-evidence / text controls, not head training or checkpoint selection.'))
    print('22 paired draft questions in 11 files; official anchor labels preserved')


if __name__ == '__main__':
    main()
