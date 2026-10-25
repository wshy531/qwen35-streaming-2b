"""Publication failure cases run without model dependencies or gated data."""
import copy
import importlib.util
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'verify_delivery.py'
spec=importlib.util.spec_from_file_location('sember_resolution_delivery',SOURCE)
delivery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


class DeliveryTests(unittest.TestCase):
    def fixture(self):
        return [dict(id='q',arm=a,resolution=r,shift=s) for a,r in delivery.CONDITIONS for s in range(5)]
    def test_complete_publication_without_private_labels(self):
        result=delivery.verify()
        self.assertEqual(result['primary_answers'],5100)
        self.assertEqual(result['replayed_answers'],270)
        self.assertFalse(result['official_labels_rescored'])
        self.assertEqual(result['writes'],0)
    def test_duplicate_cannot_replace_missing_condition(self):
        rows=self.fixture();rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):delivery.grid(rows,{'q'})
    def test_missing_rotation_is_rejected(self):
        with self.assertRaises(ValueError):delivery.grid(self.fixture()[:-1],{'q'})
    def test_replay_generated_token_drift_is_rejected(self):
        row=dict(generated_ids=[3],text='A',input_ids_sha256='i',position_ids_sha256='p',frames_sha256='f',pixels_sha256='x')
        changed=dict(row,generated_ids=[4])
        with self.assertRaisesRegex(ValueError,'generated_ids'):delivery.replay_equal(changed,row)
    def test_replay_pixel_drift_is_rejected(self):
        row=dict(generated_ids=[3],text='A',input_ids_sha256='i',position_ids_sha256='p',frames_sha256='f',pixels_sha256='x')
        with self.assertRaisesRegex(ValueError,'pixels_sha256'):delivery.replay_equal(dict(row,pixels_sha256='y'),row)
    def test_manifest_cannot_escape_publication_directory(self):
        with self.assertRaises(ValueError):delivery.safe(SOURCE.parent,'../outside.json')


if __name__=='__main__':unittest.main()
