import copy
from dataclasses import dataclass
import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.wshy.online_memory_2b.protocol import read
from experiments.wshy.sember_102_2b.inputs import base_inputs, bundle_path, check_prefix, snapshots, FIXED
from experiments.wshy.sember_102_2b.gates import Budget
from experiments.wshy.sember_102_2b import gates
from experiments.wshy.online_memory_2b.isolation import write_root
from experiments.wshy.online_memory_2b.protocol import sha
from experiments.wshy.sember_102_2b.readout import complete_question_runtime
from experiments.wshy.sember_102_2b.scoring import build, evidence_positions
from experiments.wshy.long_video_gpu_2b.scoring import scorer


class BaselineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries, cls.groups, cls.media = base_inputs()
        cls.labels = read(bundle_path() / 'labels-offline-only.json')
        cls.measure = staticmethod(scorer().measure)

    def predictions(self):
        gold = {r['id']: r['correct_letter'] for r in self.labels}
        return [dict(id=q['id'], arm=arm, status='ok', text=gold[q['id']],
                     source_path=q['source_path'], query_time_s=q['query_time_s'])
                for q in self.queries for arm in FIXED['arms']]

    def test_original102_requests_have_no_labels(self):
        self.assertEqual((len(self.queries), len(self.groups), len(self.media)), (102, 18, 18))
        for query in self.queries:
            self.assertEqual(set(query), {'id', 'source_path', 'query_time_s', 'question', 'options', 'cohort'})

    def test_continuous_queries_keep_odd_frame_pending(self):
        frames = [dict(frame_id=i, pts_s=float(i), rgb_sha256='0'*64) for i in range(10)]
        queries = [dict(self.queries[0], id='early', query_time_s=2.0),
                   dict(self.queries[0], id='late', query_time_s=7.0)]
        result, totals = snapshots(frames, queries)
        self.assertEqual((result['early']['committed_frames'], result['early']['pending_frame']['frame_id']), (2, 2))
        self.assertEqual((result['late']['committed_frames'], totals['encoded_pairs']), (8, 4))
        self.assertEqual(totals['sampled_frames'], 8)
        self.assertFalse(result['early']['query_padding'])

    def test_changed_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            check_prefix(dict(query_padding=False, committed_frames=2), dict(query_padding=False, committed_frames=4))

    def test_longer_question_allowance_preserves_other_limits(self):
        @dataclass
        class Limits:
            context_tokens: int
            visual_tokens: int
            nodes: int
            suffix_tokens: int = 128
        env = {'Limits': Limits}
        adapted = complete_question_runtime(env)
        actual = adapted['Limits'](8192, 6144, 128)
        self.assertEqual((actual.context_tokens, actual.visual_tokens, actual.nodes, actual.suffix_tokens),
                         (8192, 6144, 128, 512))
        self.assertEqual(env['Limits'](8192, 6144, 128).suffix_tokens, 128)

    def test_answer_duplicate_and_generation_cap(self):
        budget = Budget()
        row = dict(id=self.queries[0]['id'], arm='M0', generated_ids=[42], text='A')
        budget.answer(row)
        with self.assertRaises(ValueError):
            budget.answer(row)
        with self.assertRaises(ValueError):
            Budget().answer(dict(row, generated_ids=list(range(9))))

    def test_official_gold_produces_full_grid_score(self):
        summary, detail = build(self.queries, self.labels, self.predictions(), self.measure)
        self.assertEqual(len(detail), 612)
        self.assertEqual(summary['tables']['all_original102']['M0']['content_correct'], 102)
        self.assertNotIn('strict_correct', detail[0])
        self.assertNotIn('format_compliant', detail[0])
        self.assertEqual(summary['human_semantic_review'], 'pending')

    def test_duplicate_prediction_does_not_replace_missing_answer(self):
        predictions = self.predictions()
        predictions[-1] = copy.deepcopy(predictions[0])
        with self.assertRaises(ValueError):
            build(self.queries, self.labels, predictions, self.measure)

    def test_label_index_letter_mismatch_is_rejected(self):
        labels = copy.deepcopy(self.labels)
        labels[0]['correct_index'] = (labels[0]['correct_index'] + 1) % 5
        with self.assertRaises(ValueError):
            build(self.queries, labels, self.predictions(), self.measure)

    def test_paired_gains_and_losses_survive_zero_net(self):
        predictions = self.predictions()
        for index, arm in ((0, 'M0'), (1, 'M2')):
            qid = self.queries[index]['id']
            row = next(p for p in predictions if p['id'] == qid and p['arm'] == arm)
            row['text'] = 'B' if row['text'] == 'A' else 'A'
        summary, _ = build(self.queries, self.labels, predictions, self.measure)
        paired = summary['paired_comparisons']['all_original102']['M0']['M2']
        self.assertEqual(paired, dict(corrections=[self.queries[0]['id']], new_losses=[self.queries[1]['id']], net=0))

    def test_evidence_proxy_uses_committed_frames_only(self):
        frames = [dict(frame_id=i, pts_s=float(i), rgb_sha256='0'*64) for i in range(21)]
        query = dict(self.queries[0], query_time_s=20.0)
        prefixes, _ = snapshots(frames, [query])
        source = dict(frames=frames, queries=[query], expected_prefix=prefixes)
        label = dict(self.labels[0], answer_start_time=20, answer_end_time=20)
        self.assertEqual(evidence_positions([source], [label])[query['id']], 'no_committed_sample_hit')
        label.update(answer_start_time=0, answer_end_time=1)
        self.assertEqual(evidence_positions([source], [label])[query['id']], 'outside_M0_only')

    def test_prepared_gate_binds_sampling_manifest_before_execution(self):
        with tempfile.TemporaryDirectory(dir=write_root() / 'cache') as temp:
            root = Path(temp).resolve()
            self.assertTrue(root.is_relative_to(write_root()))
            run, bundle, model = root / 'run', root / 'bundle', root / 'model'
            for folder in (run, bundle / 'sampling', model):
                folder.mkdir(parents=True)
            for path in (bundle / 'sha256.json', bundle / 'sampling/manifest.json',
                         model / 'DOWNLOAD_MANIFEST.json', run / 'source_lock.json'):
                path.write_text('{}', encoding='utf-8')
            sources = [dict(frames=[{}, {}])]
            manifest_sha = sha(bundle / 'sha256.json')
            protocol = dict(gates.PROFILE, run_path=str(run), write_root=str(write_root()),
                bundle_path=str(bundle), bundle_manifest_sha256=manifest_sha,
                sampling_manifest_sha256=sha(bundle / 'sampling/manifest.json'),
                source_lock_sha256=sha(run / 'source_lock.json'), model_path=str(model),
                model_manifest_sha256=sha(model / 'DOWNLOAD_MANIFEST.json'),
                sampling_counts=dict(sampled_frames=2, encoded_pairs=1, final_pending=0))
            (run / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
            with patch.object(gates, 'bundle_path', return_value=bundle), \
                 patch.object(gates, 'MANIFEST_SHA', manifest_sha), \
                 patch.object(gates, 'source_lock', return_value={}), \
                 patch.object(gates, 'verify_model'), \
                 patch.object(gates, 'load_bundle', return_value=sources):
                self.assertEqual(gates.verify_prepared(run)[1], sources)
                (bundle / 'sampling/manifest.json').write_text('{"changed":true}', encoding='utf-8')
                with self.assertRaises(ValueError):
                    gates.verify_prepared(run)


if __name__ == '__main__':
    unittest.main()
