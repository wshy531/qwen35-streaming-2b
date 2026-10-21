"""Optional A06 policy/session contracts; no real weights or capability scoring."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

from experiments.wshy.online_memory_2b.a06 import DepthForest384
from experiments.wshy.online_memory_2b.frozen import ARMS, policies
from experiments.wshy.online_memory_2b.protocol import FIXED_PROTOCOL
from experiments.wshy.online_memory_2b.session import MemorySession, digest

FIXTURE = Path(__file__).parent/'fixtures/upstream_policy.py'
UPSTREAM_SHA = '4404e3feae920f0d26f4296718274dda328c8ff41a8d82287d636617fab28bae'


def block(i, width=32):
    values = torch.randn(48, width, generator=torch.Generator().manual_seed(100+i))
    return policies().features.raw_block(values, i, 2.*i, 2.*i+1)


def append(session, i, width=32):
    return session.append_pair(block(i, width), frame_ids=(2*i, 2*i+1))


class A06Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != UPSTREAM_SHA:
            raise ValueError('upstream A06 oracle identity changed')
        policies()  # Preload the frozen memory384 module before upstream import.
        paths = sys.path[:]
        try:
            spec = importlib.util.spec_from_file_location('_online2b_upstream_A06', FIXTURE)
            upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upstream)
        finally:
            sys.path[:] = paths
        cls.Upstream = upstream.DepthForest384

    def test_upstream_bytes_and_provenance_are_locked(self):
        info = json.loads((FIXTURE.parent/'provenance.json').read_text())
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), UPSTREAM_SHA)
        self.assertEqual(info['policy_sha256'], UPSTREAM_SHA)
        self.assertTrue(info['source_read_complete'])
        self.assertEqual(info['cluster_writes'], 0)
        self.assertEqual(info['submitted_jobs'], 0)

    def test_depth_priority_can_override_original_cost(self):
        bank = DepthForest384()
        bank.roots = [SimpleNamespace(depth=d) for d in (12, 2, 2, 12)]
        self.assertEqual(bank.select([{'total':0.}, {'total':100.}, {'total':0.}]), 1)

    def test_cost_then_index_resolve_equal_depth(self):
        bank = DepthForest384()
        bank.roots = [SimpleNamespace(depth=2) for _ in range(4)]
        self.assertEqual(bank.select([{'total':3.}, {'total':1.}, {'total':1.}]), 1)
        self.assertEqual(bank.select([{'total':1.} for _ in range(3)]), 0)

    def test_no_closed_overflow_is_bitwise_frozen_M2(self):
        for length, boundary in ((8, lambda i: True), (80, lambda i: i == 0)):
            old, candidate = policies().forest384.Forest384(), DepthForest384()
            for i in range(length):
                value = block(i)
                for bank in (old, candidate):
                    bank.append(value, starts_event=boundary(i))
                self.assertEqual(digest(old.export()), digest(candidate.export()))
            self.assertEqual(candidate.selection_count, 0)

    def test_every_prefix_matches_exact_upstream_and_original_selector_matches_M2(self):
        schedules = (lambda i: True, lambda i: i == 0,
                     lambda i: i % 7 == 0, lambda i: i % 13 in (0, 1, 4))
        selections = 0
        changed = False
        for boundary in schedules:
            old = policies().forest384.Forest384()
            compat, reference, candidate = self.Upstream(original_selector=True), self.Upstream(), DepthForest384()
            for i in range(96):
                value = block(i)
                for bank in (old, compat, reference, candidate):
                    bank.append(value, starts_event=boundary(i))
                self.assertEqual(digest(old.export()), digest(compat.export()))
                self.assertEqual(digest(reference.export()), digest(candidate.export()))
                self.assertEqual(reference.status(), candidate.status())
                self.assertEqual(reference.selection_count, candidate.selection_count)
                self.assertEqual(old.within_merges, candidate.within_merges)
                self.assertEqual(old.long_merges, candidate.long_merges)
                self.assertEqual(digest([n for n,_ in old.recent]), digest([n for n,_ in candidate.recent]))
                self.assertEqual(digest([] if old.open is None else [old.open]),
                                 digest([] if candidate.open is None else [candidate.open]))
                changed |= digest(old.export()) != digest(candidate.export())
            selections += candidate.selection_count
        self.assertGreater(selections, 0)
        self.assertTrue(changed, 'synthetic schedules must exercise a different merge choice')

    def test_session_uses_same_detector_flags_and_matches_upstream(self):
        old, session, reference = MemorySession('M2'), MemorySession('M2_A06'), self.Upstream()
        for i in range(96):
            value = block(i)
            first = old.append_pair(value, frame_ids=(2*i, 2*i+1))
            second = session.append_pair(value, frame_ids=(2*i, 2*i+1))
            self.assertEqual(first['starts_event'], second['starts_event'])
            self.assertEqual(first['adjacent_distance'], second['adjacent_distance'])
            self.assertTrue(torch.equal(old.previous, session.previous))
            reference.append(value, starts_event=first['starts_event'])
            self.assertEqual(digest(reference.export()), digest(session.export()))

    def test_512_pair_storage_mass_and_provenance_bounds_without_archive(self):
        session = MemorySession('M2_A06')
        expected_fields = set(policies().forest384.Forest384().__dict__) | {'selection_count'}
        for i in range(512):
            append(session, i)
            state, nodes = session.status(), session.export()
            self.assertEqual(set(session.bank.__dict__), expected_fields)
            self.assertLessEqual(state['visual_tokens'], 384)
            self.assertLessEqual(state['resident_bytes'], 8*7488+128)
            self.assertLessEqual(state['metadata_bytes'], 32768)
            self.assertLessEqual(len(nodes), 8)
            self.assertLessEqual(len(session.bank.roots), 3)
            self.assertEqual(nodes[0].first, 0)
            self.assertEqual(nodes[-1].last, i)
            self.assertTrue(all(a.last+1 == b.first for a,b in zip(nodes,nodes[1:])))
            self.assertAlmostEqual(sum(float(n.mass.sum()) for n in nodes), 48*(i+1), delta=1e-3)
            self.assertFalse(state['full_history_retained'])
            self.assertFalse(state['language_cache_retained'])

    def test_real_2b_feature_width_counts_descriptor_and_budget(self):
        session = MemorySession('M2_A06')
        for i in range(16):
            append(session, i, 2048)
        self.assertEqual(session.status()['detector_bytes'], 8192)
        self.assertLessEqual(session.status()['resident_bytes'], 384*(2048*4+4+3*8)+8192)

    def test_answers_do_not_change_continuation_or_torch_RNG(self):
        asked = MemorySession('M2_A06', lambda ns,q: dict(text='A', generated_ids=[1]))
        control = MemorySession('M2_A06')
        for i in range(48):
            append(asked, i)
            append(control, i)
            rng = torch.get_rng_state().clone()
            self.assertTrue(asked.ask('q')['memory_unchanged'])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(asked.state_digest(), control.state_digest())

    def test_export_and_reader_input_mutation_do_not_change_bank(self):
        session = MemorySession('M2_A06')
        for i in range(12):
            append(session, i)
        before = session.state_digest()
        session.export()[0].features.zero_()
        self.assertEqual(before, session.state_digest())
        def mutate(nodes, question):
            nodes[0].features.zero_()
            return dict(text='A')
        session.readout = mutate
        with self.assertRaisesRegex(RuntimeError, 'exported input'):
            session.ask('q')
        self.assertEqual(before, session.state_digest())

    def test_reader_exception_preserves_state(self):
        session = MemorySession('M2_A06')
        append(session, 0)
        before = session.state_digest()
        def fail(nodes, question):
            raise ValueError('reader failed')
        session.readout = fail
        with self.assertRaisesRegex(ValueError, 'reader failed'):
            session.ask('q')
        self.assertEqual(before, session.state_digest())

    def test_descriptor_and_selection_counter_mutations_are_detected(self):
        for field in ('descriptor', 'selection_counter', 'event_flags'):
            session = MemorySession('M2_A06')
            append(session, 0)
            def mutate(nodes, question):
                if field == 'descriptor':
                    session.previous.add_(1)
                elif field == 'selection_counter':
                    session.bank.selection_count += 1
                else:
                    node, flag = session.bank.recent[0]
                    session.bank.recent[0] = node, not flag
                return dict(text='A')
            session.readout = mutate
            with self.assertRaisesRegex(RuntimeError, 'persistent memory'):
                session.ask('q')
            self.assertFalse(session.valid)
            with self.assertRaises(RuntimeError):
                append(session, 1)
            session.reset()
            self.assertEqual(session.state_digest(), MemorySession('M2_A06').state_digest())

    def test_invalid_pairs_rejected_before_mutation(self):
        session = MemorySession('M2_A06')
        append(session, 0)
        before = session.state_digest()
        bad_time = session.legacy.features.raw_block(torch.zeros(48,32), 1, 1., 2.)
        bad_number = block(1)
        bad_number.features[0,0] = float('nan')
        cases = ((block(0), (2,3)), (block(2), (2,3)), (block(1,64), (2,3)),
                 (block(1), (1,2)), (block(1), (2,2)), (bad_time, (2,3)), (bad_number, (2,3)))
        for value, frames in cases:
            with self.assertRaises(ValueError):
                session.append_pair(value, frame_ids=frames)
            self.assertTrue(session.valid)
            self.assertEqual(session.state_digest(), before)

    def test_invalid_direct_event_flag_is_atomic(self):
        bank = DepthForest384()
        bank.append(block(0), starts_event=True)
        before = digest(bank.export())
        with self.assertRaises(ValueError):
            bank.append(block(1), starts_event='yes')
        self.assertEqual(digest(bank.export()), before)

    def test_odd_tail_reset_and_close_release_all_A06_state(self):
        session = MemorySession('M2_A06')
        for i in range(20):
            append(session, i)
        self.assertGreater(session.bank.selection_count, 0)
        value = session.legacy.features.raw_block(torch.zeros(48,32), 20, 40., 40.)
        session.append_pair(value, frame_ids=(40,40), padded_tail=True)
        with self.assertRaises(ValueError):
            append(session, 21)
        session.reset()
        self.assertEqual(session.bank.selection_count, 0)
        self.assertIsNone(session.previous)
        self.assertEqual(session.status()['resident_bytes'], 0)
        self.assertEqual(session.state_digest(), MemorySession('M2_A06').state_digest())
        append(session, 0)
        session.close()
        self.assertEqual(session.status()['resident_bytes'], 0)
        with self.assertRaises(RuntimeError):
            append(session, 0)
        with self.assertRaises(RuntimeError):
            session.ask('q')
        session.reset()
        append(session, 0)

    def test_candidate_does_not_enter_default_replay_protocol(self):
        self.assertEqual(ARMS, ('M0','M2','reservoir4','text_only'))
        self.assertEqual(FIXED_PROTOCOL['arms'], list(ARMS))
        self.assertEqual(FIXED_PROTOCOL['main_answers'], 684)
        self.assertEqual(FIXED_PROTOCOL['maximum_answers'], 756)


if __name__ == '__main__':
    unittest.main()
