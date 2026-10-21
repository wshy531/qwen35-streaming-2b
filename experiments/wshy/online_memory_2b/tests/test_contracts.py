import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.wshy.online_memory_2b.frozen import NATURAL, module, policies
from experiments.wshy.online_memory_2b.protocol import safe_path, validate_requests
from experiments.wshy.online_memory_2b.session import MemorySession, digest
from experiments.wshy.online_memory_2b.stream import decode_pairs


def block(i, width=32):
    return policies().features.raw_block(torch.randn(48,width,
        generator=torch.Generator().manual_seed(100+i)),i,2.*i,2.*i+1)


def append(session, i, width=32):
    return session.append_pair(block(i,width),frame_ids=(2*i,2*i+1))


class SessionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_frozen_policy_replay_at_many_prefixes(self):
        legacy=policies()
        nodes=[legacy.forest.raw(block(i)) for i in range(70)]
        fixed=module('fixed_stage2',NATURAL)
        old_diag=fixed.DIAG
        fixed.DIAG=NATURAL/'diag'
        try:
            for length in (1,4,8,9,19,70):
                exports,_,_=fixed.policies({'load':lambda:legacy},nodes[:length])
                for arm in ('M0','M2'):
                    session=MemorySession(arm)
                    for i in range(length):
                        append(session,i)
                    self.assertEqual(digest(session.export()),digest(exports[arm]))
        finally:
            fixed.DIAG=old_diag

    def test_reservoir_replays_fixed_seed_and_recent_pairs(self):
        legacy=policies()
        reference=legacy.reservoir.ReservoirRecent()
        session=MemorySession('reservoir4')
        for i in range(90):
            reference.append(legacy.forest.raw(block(i)))
            append(session,i)
            self.assertEqual(digest(reference.export()),digest(session.export()))
        self.assertEqual([n.first for n in session.export()][-4:],[86,87,88,89])

    def test_long_sequences_remain_bounded_without_hidden_feature_history(self):
        for arm in ('M0','M2','reservoir4'):
            session=MemorySession(arm)
            for i in range(512):
                append(session,i)
                state=session.status()
                self.assertLessEqual(state['visual_tokens'],384)
                self.assertLessEqual(state['resident_bytes'],8*7488+128)
                self.assertLessEqual(len(session._nodes()),8)
                self.assertFalse(state['full_history_retained'])
                self.assertFalse(state['language_cache_retained'])

    def test_invalid_append_is_atomic(self):
        session=MemorySession('M2')
        append(session,0)
        before=session.state_digest()
        for invalid in (lambda:append(session,0),lambda:append(session,2),
                        lambda:append(session,1,64),
                        lambda:session.append_pair(block(1),frame_ids=(1,2)),
                        lambda:session.append_pair(block(1),frame_ids=(2,2))):
            with self.assertRaises(ValueError):
                invalid()
            self.assertEqual(before,session.state_digest())
            self.assertTrue(session.valid)

    def test_nonfinite_and_overlapping_timestamps_rejected(self):
        session=MemorySession('M0')
        append(session,0)
        value=block(1)
        value.features[0,0]=float('nan')
        with self.assertRaises(ValueError):
            session.append_pair(value,frame_ids=(2,3))
        value=session.legacy.features.raw_block(torch.zeros(48,32),1,1.,2.)
        with self.assertRaises(ValueError):
            session.append_pair(value,frame_ids=(2,3))

    def test_odd_tail_is_terminal_until_reset(self):
        session=MemorySession('M0')
        append(session,0)
        value=session.legacy.features.raw_block(torch.zeros(48,32),1,2.,2.)
        session.append_pair(value,frame_ids=(2,2),padded_tail=True)
        self.assertEqual(session.append_count,2)
        self.assertEqual(session.export()[-1].begin,session.export()[-1].end)
        with self.assertRaises(ValueError):
            append(session,2)

    def test_ask_isolation_and_continuation_match_never_asked(self):
        for arm in ('M0','M2','reservoir4'):
            asked=MemorySession(arm,lambda ns,q:dict(text='A',generated_ids=[1]))
            control=MemorySession(arm)
            for i in range(25):
                append(asked,i)
                append(control,i)
                self.assertTrue(asked.ask('q')['memory_unchanged'])
                self.assertEqual(asked.state_digest(),control.state_digest())

    def test_export_cannot_alias_persistent_storage(self):
        session=MemorySession('M2')
        for i in range(12):
            append(session,i)
        before=session.state_digest()
        exported=session.export()
        exported[0].features.zero_()
        self.assertEqual(before,session.state_digest())

    def test_readout_exception_preserves_bank(self):
        def fail(nodes,q):
            raise ValueError('test reader failed')
        session=MemorySession('M0',fail)
        append(session,0)
        before=session.state_digest()
        with self.assertRaises(ValueError):
            session.ask('q')
        self.assertEqual(before,session.state_digest())

    def test_readout_input_mutation_detected_and_bank_preserved(self):
        def mutate(nodes,q):
            nodes[0].features.zero_()
            return dict(text='A')
        session=MemorySession('M0',mutate)
        append(session,0)
        before=session.state_digest()
        with self.assertRaisesRegex(RuntimeError,'exported input'):
            session.ask('q')
        self.assertEqual(before,session.state_digest())

    def test_persistent_bank_mutation_invalidates_session(self):
        session=MemorySession('M0')
        append(session,0)
        def mutate(nodes,q):
            session.bank.recent[0].payload.features.zero_()
            return dict(text='A')
        session.readout=mutate
        with self.assertRaisesRegex(RuntimeError,'persistent memory'):
            session.ask('q')
        self.assertFalse(session.valid)
        with self.assertRaises(RuntimeError):
            append(session,1)
        session.reset()
        append(session,0)

    def test_reset_and_close_release_all_policy_state(self):
        for arm in ('M0','M2','reservoir4'):
            session=MemorySession(arm)
            for i in range(20):
                append(session,i)
            session.reset()
            self.assertEqual(session.state_digest(),MemorySession(arm).state_digest())
            self.assertEqual(session.status()['resident_bytes'],0)
            append(session,0)
            session.close()
            self.assertEqual(session.status()['resident_bytes'],0)
            with self.assertRaises(RuntimeError):
                append(session,0)
            with self.assertRaises(RuntimeError):
                session.ask('q')
            session.reset()
            append(session,0)

    def test_question_cannot_silently_change_reservoir_rng(self):
        session=MemorySession('reservoir4')
        append(session,0)
        def mutate(nodes,q):
            session.bank.rng.random()
            return dict(text='A')
        session.readout=mutate
        with self.assertRaisesRegex(RuntimeError,'persistent memory'):
            session.ask('q')
        self.assertFalse(session.valid)

    def test_2b_width_and_m2_boundary_replay(self):
        session=MemorySession('M2')
        events=[]
        for i in range(10):
            value=session.legacy.features.raw_block(torch.full((48,2048),1. if i<5 else -1.),i,2.*i,2.*i+1)
            events.append(session.append_pair(value,frame_ids=(2*i,2*i+1)))
        self.assertTrue(events[0]['starts_event'])
        self.assertFalse(events[4]['starts_event'])
        self.assertTrue(events[5]['starts_event'])
        self.assertEqual(session.status()['detector_bytes'],2048*4)


class InputContracts(unittest.TestCase):
    def request(self):
        return dict(id='q',cohort='c',media_path='m',media_sha256='hash',model_visible=dict(
            question='q',options=['a','b'],query_time_s=3.,frames=[
                dict(frame_id=i,pts_s=float(i),rgb_sha256='hash') for i in range(4)]))

    def test_requests_reject_future_duplicate_and_labels(self):
        validate_requests([self.request()])
        for change in ('future','duplicate','label','frame_order'):
            row=self.request()
            if change=='future':
                row['model_visible']['frames'][-1]['pts_s']=4.
            elif change=='label':
                row['model_visible']['answer_index']=1
            elif change=='frame_order':
                row['model_visible']['frames'][1]['frame_id']=0
            with self.assertRaises(ValueError):
                validate_requests([row,row] if change=='duplicate' else [row])

    def test_paths_refuse_escape_and_cross_platform_paths(self):
        root=Path.cwd()
        for relative in ('../outside','/outside','C:/outside','a\\b'):
            with self.assertRaises(ValueError):
                safe_path(root,relative)

    def test_decoder_yields_before_next_pair_and_never_reads_after_prefix(self):
        decoded=[]
        class Container:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            streams=SimpleNamespace(video=[SimpleNamespace(codec_context=SimpleNamespace())])
            def decode(self,stream):
                for i in range(10):
                    decoded.append(i)
                    yield SimpleNamespace(time=float(i))
        rgb=bytes(192*256*3)
        env=dict(av=SimpleNamespace(open=lambda p:Container()),np=np,
                 canvas_bytes=lambda f:rgb,FramePacket=lambda *args:args)
        frames=[dict(frame_id=i,pts_s=float(i),rgb_sha256=hashlib.sha256(rgb).hexdigest()) for i in range(3)]
        iterator=decode_pairs(env,'unused',frames,2.)
        first,padded,_=next(iterator)
        self.assertEqual(decoded,[0,1])
        self.assertFalse(padded)
        tail,padded,_=next(iterator)
        self.assertEqual(decoded,[0,1,2])
        self.assertTrue(padded)
        self.assertIs(tail[0],tail[1])
        with self.assertRaises(StopIteration):next(iterator)
        self.assertEqual(decoded,[0,1,2])


if __name__=='__main__':
    unittest.main()
