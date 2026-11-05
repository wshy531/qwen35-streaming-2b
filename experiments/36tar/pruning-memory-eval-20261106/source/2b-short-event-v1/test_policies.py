"""Bounded CPU contract tests; no pretrained weights and no capability claims."""
import hashlib
import unittest

import bridge
import torch
from legacy_sources import load
from memory384 import Forest384
from policies import CappedForest, ReservoirRecent, block
from scoring import measure


def node(i, width=32):
    legacy = load()
    return legacy.forest.raw(legacy.features.raw_block(
        torch.randn(48, width, generator=torch.Generator().manual_seed(100+i)),
        i, 2.0*i, 2.0*i+1))


def digest(nodes):
    h = hashlib.sha256()
    for n in nodes:
        h.update(str(n.metadata()).encode())
        for t in (n.features, n.mass, n.coords):
            h.update(t.numpy().tobytes())
    return h.hexdigest()


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unbounded_candidate_replays_existing_m2(self):
        original, candidate = Forest384(), CappedForest(10000)
        for i in range(90):
            n = node(i)
            flag = i % 7 == 0
            original.append(block(n), starts_event=flag)
            candidate.append(block(n), starts_event=flag)
            self.assertEqual(digest(original.export()), digest(candidate.export()))
            for field in ('within_merges', 'long_merges', 'tokens', 'nodes'):
                self.assertEqual(original.status()[field], candidate.status()[field])

    def test_forced_seals_and_budget_at_each_write(self):
        bank = CappedForest()
        for i in range(512):
            bank.append(block(node(i)), starts_event=i == 0)
            s = bank.status()
            self.assertLessEqual(s['tokens'], 384)
            self.assertLessEqual(s['closed_tokens'], 144)
            self.assertLessEqual(s['open_pair_span'], 4)
            self.assertEqual([n.first for n, _ in bank.recent], list(range(max(0, i-3), i+1)))
            self.assertEqual(s['represented_mass'], 48*(i+1))
        self.assertGreater(bank.forced_seals, 0)
        self.assertGreater(bank.long_merges, 0)

    def test_true_boundaries_remain_effective(self):
        bank = CappedForest()
        for i in range(32):
            bank.append(block(node(i)), starts_event=True)
        self.assertEqual(bank.forced_seals, 0)
        self.assertEqual(bank.status()['open_pair_span'], 1)

    def test_export_owns_storage_and_past_snapshots_survive(self):
        bank = CappedForest()
        for i in range(12):
            bank.append(block(node(i)), starts_event=i % 5 == 0)
        snapshot = bank.export()
        before = digest(snapshot)
        for i in range(12, 20):
            bank.append(block(node(i)), starts_event=False)
        self.assertEqual(before, digest(snapshot))
        current = digest(bank.export())
        snapshot[0].features.zero_()
        self.assertEqual(current, digest(bank.export()))

    def test_2b_width_and_input_rejection(self):
        bank = CappedForest()
        for i in range(10):
            bank.append(block(node(i, 2048)), starts_event=i == 0)
        self.assertTrue(all(n.features.shape[1] == 2048 for n in bank.export()))
        self.assertLessEqual(bank.status()['resident_tensor_bytes'], 3156480)
        with self.assertRaises(ValueError):
            bank.append(block(node(10, 32)), starts_event=False)
        with self.assertRaises(ValueError):
            bank.append(block(node(11, 2048)), starts_event=False)
        with self.assertRaises(ValueError):
            bank.append(block(node(10, 2048)), starts_event=1)

    def test_reservoir_is_bounded_reproducible_and_keeps_recent(self):
        a, b = ReservoirRecent(), ReservoirRecent()
        for i in range(512):
            n = node(i)
            a.append(n)
            b.append(n)
            self.assertLessEqual(len(a.old), 4)
            self.assertLessEqual(len(a.recent), 4)
            self.assertEqual([n.first for n in a.recent], list(range(max(0, i-3), i+1)))
            self.assertEqual(digest(a.export()), digest(b.export()))
        self.assertEqual(a.promoted, 508)
        self.assertEqual(a.status()['tokens'], 384)
        self.assertEqual(len({n.first for n in a.export()}), 8)

    def test_prespecified_content_and_format_are_distinct(self):
        for text in ('C', 'c.', ' C! '):
            s = measure(text, 5, 2)
            self.assertTrue(s['strict_correct'] and s['content_correct'])
        s = measure('C. close the lid', 5, 2)
        self.assertTrue(s['content_correct'])
        self.assertFalse(s['strict_correct'])
        for text in ('Cat', 'The answer is C', '', 'E', 'B or C'):
            s = measure(text, 4, 2)
            self.assertFalse(s['content_correct'])
        self.assertFalse(measure('D', 3, 2)['content_parseable'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
