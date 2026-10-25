import copy
import unittest

from experiments.wshy.sember_102_2b.readout import question_prompt, complete_question_runtime
from experiments.wshy.sember_102_2b.inputs import base_inputs, bundle_path
from experiments.wshy.online_memory_2b.protocol import read
from experiments.wshy.sember_readout_diagnostic.plan import requests, select, PROFILE, ARMS
from experiments.wshy.sember_readout_diagnostic.scoring import build


class PromptContracts(unittest.TestCase):
    def test_all102_official_options_render_once_without_mutation(self):
        queries, _, _ = base_inputs()
        before = copy.deepcopy(queries)
        for q in queries:
            text = question_prompt(q)
            expected = q['question']+'\n'+'\n'.join(q['options'])+'\nReturn only the single option letter. Do not explain.'
            self.assertEqual(text, expected)
        self.assertEqual(queries, before)

    def test_unlabelled_and_labelled_same_content(self):
        self.assertEqual(question_prompt(dict(question='Q', options=['cat', 'dog'])),
                         question_prompt(dict(question='Q', options=['A. cat', 'B. dog'])))

    def test_content_prefix_not_recursively_removed(self):
        self.assertIn('A. A. Smith', question_prompt(dict(question='Q', options=['A. A. Smith', 'B. Brown'])))

    def test_mixed_labels_fail(self):
        with self.assertRaises(ValueError):
            question_prompt(dict(question='Q', options=['A. cat', 'dog']))

    def test_reordered_labels_fail(self):
        with self.assertRaises(ValueError):
            question_prompt(dict(question='Q', options=['B. cat', 'A. dog']))

    def test_runtime_installs_corrected_prompt_without_changing_parent(self):
        old = lambda q: 'legacy'
        env = dict(Limits=lambda **kw: kw, prompt=old)
        adapted = complete_question_runtime(env)
        self.assertIs(adapted['prompt'], question_prompt)
        self.assertIs(env['prompt'], old)


class EvidenceContracts(unittest.TestCase):
    def frames(self, count=10):
        return [dict(frame_id=30*i, pts_s=float(i), rgb_sha256=f'{i:064x}') for i in range(count)]

    def test_interval_keeps_global_pair_boundary(self):
        result = select(self.frames(), 8, [3, 4])
        self.assertEqual(result['evidence_pair_indices'], [1, 2])
        self.assertEqual([f['pts_s'] for f in result['evidence']], [2., 3., 4., 5.])

    def test_pending_evidence_is_not_padded_or_used(self):
        result = select(self.frames(), 8, [8, 8])
        self.assertEqual(len(result['prefix']), 8)
        self.assertEqual(result['evidence_status'], 'no_committed_evidence_sample')
        self.assertEqual(result['pending_frame']['pts_s'], 8)

    def test_no_future_frame(self):
        result = select(self.frames(), 3.5, [0, 3.5])
        self.assertTrue(all(f['pts_s'] <= 3.5 for f in result['prefix']))
        self.assertEqual(len(result['evidence']), 4)

    def test_future_annotation_rejected(self):
        with self.assertRaises(ValueError):
            select(self.frames(), 4, [0, 5])

    def test_no_budget_thinning(self):
        with self.assertRaises(ValueError):
            select(self.frames(602), 601, [0, 601])

    def test_all102_plan_excludes_answers(self):
        q, _, _ = base_inputs()
        result = requests(q, read(bundle_path() / 'labels-offline-only.json'))
        self.assertEqual(len(result), 102)
        for r in result:
            self.assertEqual(set(r), {'query', 'evidence_interval', 'prompt_sha256'})
            self.assertNotIn('correct_index', r['query'])
        self.assertEqual(PROFILE['maximum_answers'], len(result)*len(ARMS))

    def test_wrong_video_annotation_rejected(self):
        q, _, _ = base_inputs()
        labels = read(bundle_path() / 'labels-offline-only.json')
        labels[0]['video_id'] = 'wrong'
        with self.assertRaises(ValueError):
            requests(q, labels)


class ScoringContracts(unittest.TestCase):
    def sample(self):
        q = dict(id='test', source_path='a.mp4', query_time_s=4, options=['x']*5)
        label = dict(id='test', correct_index=1, correct_letter='B', question_category='test')
        rows = [dict(id='test', source_path='a.mp4', query_time_s=4, arm=a, text='B. explanation', status='ok') for a in ARMS]
        old = [dict(rows[0], arm='text_only', text='A')]
        return [q], [label], rows, old

    def test_explanation_is_not_penalized(self):
        summary, detail = build(*self.sample())
        self.assertEqual(summary['totals']['current_evidence']['content_correct'], 1)
        self.assertEqual(summary['paired']['text_only_clean_vs_text_only_legacy']['net'], 1)
        self.assertNotIn('strict_correct', str(detail))

    def test_unparsed_kept_unresolved(self):
        q, labels, rows, old = self.sample()
        rows[0]['text'] = 'Not enough information'
        summary, _ = build(q, labels, rows, old)
        self.assertEqual(summary['totals']['native_prefix']['unresolved'], 1)
        self.assertEqual(summary['paired']['native_prefix_vs_text_only_clean']['evaluable_questions'], 0)

    def test_duplicate_not_counted(self):
        q, labels, rows, old = self.sample()
        with self.assertRaises(ValueError):
            build(q, labels, rows+[rows[0]], old)

    def test_source_drift_rejected(self):
        q, labels, rows, old = self.sample()
        rows[0]['source_path'] = 'b.mp4'
        with self.assertRaises(ValueError):
            build(q, labels, rows, old)


if __name__ == '__main__':
    unittest.main()
