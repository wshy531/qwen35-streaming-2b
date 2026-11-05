"""Actual local release consistency tests; no model accuracy assertions."""
import hashlib
import json
from pathlib import Path
import unittest

from build_source_registry import resolve
from evaluation_gate import require_frozen
from score_memory_eval import digest

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/dataset_build_20261018_01'
RELEASE=RUN/'content_release_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


class DatasetReleaseTests(unittest.TestCase):
    def test_all_original_questions_answers_and_splits_are_unchanged(self):
        rows=read(ROOT/'experiments/2b-memory-eval/data/candidate_manifest.json')
        actual=read(RELEASE/'question_annotation_overlay.json')['questions']
        self.assertEqual([q['original'] for q in actual],rows)
        self.assertEqual(len(actual),235)
        for q in actual:self.assertEqual(q['original_row_sha256'],digest(q['original']))

    def test_actual_witnesses_bind_to_source_and_existing_preview_files(self):
        actual=read(RELEASE/'question_annotation_overlay.json')['questions']
        for q in actual:
            if q['original']['split']!='comparison':continue
            self.assertTrue(q['review']['witnesses'],q['id'])
            for w in q['review']['witnesses']:
                self.assertEqual(w.get('source_path'),q['original']['source_path'],q['id'])
                self.assertLessEqual(w['pts_s'],q['original']['query_time_s'])
                self.assertTrue((ROOT/w['image']).is_file())
                self.assertEqual(len(w['rgb_sha256']),64)

    def test_holdout_and_unresolved_parents_are_not_promoted_to_formal(self):
        actual=read(RELEASE/'question_annotation_overlay.json')['questions']
        holdout=[q for q in actual if q['original']['split']=='holdout']
        self.assertEqual(len(holdout),94)
        for q in holdout:
            self.assertEqual(q['review']['verdict'],'not_visually_reviewed_holdout')
            self.assertFalse(q['formal_evaluation_eligible'])
        unknown=[s for s in read(RELEASE/'source_registry.json')['sources'] if s['parent_identity_status']!='verified']
        self.assertEqual(len(unknown),10)
        self.assertTrue(all(s['source_group_id'] is None for s in unknown))
        with self.assertRaises(ValueError):require_frozen(read(RELEASE/'summary.json'))

    def test_recording_grouping_and_hirest_offset_require_author_confirmation(self):
        result=resolve('ovo','hirest/abcdefghijk_81_236.mp4',{},set(),{}, {'abcdefghijk':[[81,236]]})
        self.assertEqual(result['source_group_id'],'youtube:abcdefghijk')
        self.assertEqual(result['parent_trim_s'],[81,236])
        with self.assertRaises(ValueError):resolve('ovo','hirest/abcdefghijk_81_236.mp4',{},set(),{}, {'abcdefghijk':[[0,155]]})
        unknown=resolve('ovo','Ego4D/clips/some_uuid.mp4',{},set(),{})
        self.assertEqual(unknown['parent_identity_status'],'unresolved')
        self.assertIsNone(unknown['source_group_id'])

    def test_hashes_cover_all_release_artifacts(self):
        expected=read(RELEASE/'content_hashes.json')
        actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RELEASE.iterdir() if p.is_file() and p.name!='content_hashes.json'}
        self.assertEqual(expected,actual)


if __name__=='__main__':unittest.main(verbosity=2)
