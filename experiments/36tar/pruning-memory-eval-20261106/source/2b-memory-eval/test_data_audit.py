"""Detect future-answer probes and source leakage before model comparisons."""
import unittest
from audit_development_evidence import crosstask_question, leakage_groups, source_identity, no_evidence_option_rule


class DataAuditTests(unittest.TestCase):
    def fixture(self):
        row = dict(source_path='cross_task/abcdefghijk.mp4',question='What does the person do after stir',
                   options=['serve','add onion'],answer_index=0,query_time_s=30.)
        annotations = {'abcdefghijk':dict(member='author.csv',steps=[
            dict(action='add onion',start_s=10.,end_s=15.),
            dict(action='stir',start_s=30.,end_s=35.),
            dict(action='serve',start_s=40.,end_s=45.)])}
        return row,annotations

    def test_future_answer_does_not_become_past_evidence(self):
        row,anno = self.fixture()
        result = crosstask_question(row,anno)
        self.assertEqual(result['status'],'future_answer_evidence_not_a_past_recall_probe')
        self.assertEqual(result['earliest_answer_start_minus_query_s'],10.)
        self.assertEqual(row['query_time_s'],30.)
        self.assertTrue(result['anchor_start_matches_query'])

    def test_before_and_ongoing_are_distinct(self):
        row,anno = self.fixture()
        row.update(question='What does the person do before stir',answer_index=1)
        self.assertEqual(crosstask_question(row,anno)['status'],
                         'source_supports_past_answer_visual_review_still_required')
        row.update(query_time_s=12.)
        self.assertEqual(crosstask_question(row,anno)['status'],
                         'answer_action_ongoing_at_query_requires_task_review')

    def test_same_parent_across_folders_and_clips_blocks_split(self):
        rows = [dict(id='a',dataset='ovo',source_path='COIN/abcdefghijk.mp4',split='development'),
                dict(id='b',dataset='ovo',source_path='hirest/abcdefghijk_20_90.mp4',split='holdout')]
        conflicts = leakage_groups(rows)
        self.assertEqual(list(conflicts),['youtube:abcdefghijk'])
        rows[1]['split'] = 'development'
        self.assertEqual(leakage_groups(rows),{})

    def test_unique_ego_clip_does_not_prove_parent_identity(self):
        key,status = source_identity('ovo','Ego4D/clips/unique-uuid.mp4')
        self.assertEqual(status,'parent_recording_unresolved')
        self.assertNotIn('parent:',key)

    def test_refusal_option_shortcut_requires_no_video_and_is_not_first_option(self):
        self.assertEqual(no_evidence_option_rule(dict(options=['floor','shelf','Unable to answer'])),2)
        self.assertIsNone(no_evidence_option_rule(dict(options=['floor','shelf'])))
        self.assertIsNone(no_evidence_option_rule(dict(options=['Unable to answer','unable to answer.'])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
