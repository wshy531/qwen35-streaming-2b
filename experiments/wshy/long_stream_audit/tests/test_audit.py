"""Offline scoring failures must remain visible before any ability claim."""
import copy
import unittest

from experiments.wshy.long_stream_audit.audit import build, validate_grid


def fixture():
    ids = [f'ovo:{i}' for i in range(28)] + [f'local_longstream_v1:{i:02d}' for i in range(1, 7)]
    rows = [dict(id=q, answers={a: 'A' for a in ('M0', 'M2', 'A06')},
                 generated_ids={a: [32, 248046] for a in ('M0', 'M2', 'A06')}) for q in ids]
    rows[1]['answers']['A06'] = 'B'
    rows[1]['generated_ids']['A06'] = [33, 248046]
    capture = [dict(kind='answer_grid', queries=34, rows=rows)]
    capture.extend(dict(kind='comparison', reference=arm, compared=34, baseline_replay_count=34,
                        baseline_id_mismatches=[], baseline_frame_mismatches=[],
                        A06_changed_ids=['ovo:1']) for arm in ('M0', 'M2'))
    metadata = dict(changed_cases=[dict(id='ovo:1', source_path='clip.mp4', query_time_s=12,
                                      question='Where?', options=['here', 'there', 'other', 'none'],
                                      M2=dict(text='A'), A06=dict(text='B'))])
    labels = [dict(kind='upstream_matching_questions', records=[dict(request_id='ovo:1', record=dict(
        id=1, video='clip.mp4', realtime=12, question='Where?',
        options=['here', 'there', 'other', 'none'], answer='there', gt=1))])]
    return capture, labels, metadata


class AuditContracts(unittest.TestCase):
    def test_missing_pair_is_rejected(self):
        capture, _, _ = fixture()
        capture[0]['rows'][0]['answers'].pop('M0')
        with self.assertRaisesRegex(ValueError, 'complete M0/M2/A06'):
            validate_grid(capture)

    def test_incomplete_or_mismatched_baseline_replay_is_rejected(self):
        for field, value in (('baseline_replay_count', 33), ('baseline_frame_mismatches', ['ovo:1']),
                             ('A06_changed_ids', [])):
            capture, _, _ = fixture()
            capture[1][field] = value
            with self.assertRaisesRegex(ValueError, 'receipt mismatch'):
                validate_grid(capture)

    def test_label_requires_matching_request_identity(self):
        for field, value in (('video', 'other.mp4'), ('realtime', 13), ('question', 'When?'),
                             ('options', ['there', 'here', 'other', 'none'])):
            capture, labels, metadata = fixture()
            labels[0]['records'][0]['record'][field] = value
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                build(capture, labels, metadata)

    def test_invalid_and_duplicate_labels_are_rejected(self):
        capture, labels, metadata = fixture()
        duplicate = copy.deepcopy(labels)
        duplicate[0]['records'].append(copy.deepcopy(duplicate[0]['records'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build(capture, duplicate, metadata)
        labels[0]['records'][0]['record']['gt'] = True
        with self.assertRaisesRegex(ValueError, 'invalid original label'):
            build(capture, labels, metadata)

    def test_unknown_truth_and_unparseable_answer_are_not_counted_as_correct(self):
        capture, labels, metadata = fixture()
        capture[0]['rows'][0]['answers']['M0'] = 'unknown'
        summary, details, _ = build(capture, labels, metadata)
        unknown = next(r for r in details if r['id'] == 'ovo:0')
        for arm in ('M0', 'M2', 'A06'):
            self.assertIsNone(unknown['arms'][arm]['strict_correct'])
            self.assertIsNone(unknown['arms'][arm]['content_correct'])
        self.assertIsNone(summary['overall_accuracy'])
        self.assertEqual(summary['changed_original_label_corrections'], ['ovo:1'])

    def test_strict_and_leading_choice_are_kept_separate(self):
        capture, labels, metadata = fixture()
        capture[0]['rows'][0]['answers']['M0'] = 'A. black'
        _, details, _ = build(capture, labels, metadata)
        result = next(r for r in details if r['id'] == 'ovo:0')['arms']['M0']
        self.assertFalse(result['format_compliant'])
        self.assertTrue(result['content_parseable'])

    def test_human_review_queue_excludes_predictions_and_original_labels(self):
        capture, labels, metadata = fixture()
        _, _, review = build(capture, labels, metadata)
        self.assertEqual(len(review), 34)
        for row in review:
            self.assertEqual(row['status'], 'pending')
            self.assertEqual(row['admission_group'], 'pending')
            self.assertIsNone(row['manual_label'])
            for key in ('arms', 'text', 'answer', 'generated_ids', 'original_label_index', 'correct'):
                self.assertNotIn(key, row)

    def test_token_budget_violation_and_duplicate_question_are_rejected(self):
        for mode in ('tokens', 'duplicate'):
            capture, _, _ = fixture()
            if mode == 'tokens':
                capture[0]['rows'][0]['generated_ids']['M0'] = [32]*9
            else:
                capture[0]['rows'][0]['id'] = capture[0]['rows'][1]['id']
            with self.assertRaises(ValueError):
                validate_grid(capture)


if __name__ == '__main__':
    unittest.main()
