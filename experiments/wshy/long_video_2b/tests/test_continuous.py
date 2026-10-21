import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.wshy.long_video_2b.protocol import (
    ARMS, bind_official_labels, pending_catalog, validate_queries)
from experiments.wshy.long_video_2b.scheduler import decode_samples, replay_source, validate_frames
from experiments.wshy.long_video_2b.driver import run_sessions
from experiments.wshy.online_memory_2b.frozen import NATURAL, module, policies
from experiments.wshy.online_memory_2b.session import MemorySession, digest


def query(time, qid='q', source='video.mp4'):
    return dict(id=qid, source_path=source, query_time_s=time, question='Which option?',
                options=['first', 'second'], cohort='test')


def specs(count):
    return [dict(frame_id=i, pts_s=float(i), rgb_sha256=hashlib.sha256(str(i).encode()).hexdigest())
            for i in range(count)]


class Samples:
    def __init__(self, frames):
        self.frames, self.index, self.closed = frames, 0, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.frames):
            raise StopIteration
        frame = self.frames[self.index]
        self.index += 1
        return SimpleNamespace(frame_id=frame['frame_id'], timestamp=frame['pts_s'])

    def close(self):
        self.closed = True


class QueryContracts(unittest.TestCase):
    def test_chronological_sort_is_stable_for_same_time_and_does_not_modify_input(self):
        original = [query(5, 'last'), query(2, 'first'), query(2, 'second')]
        before = copy.deepcopy(original)
        ordered = validate_queries(original)['video.mp4']
        self.assertEqual([q['id'] for q in ordered], ['first', 'second', 'last'])
        ordered[0]['options'][0] = 'changed'
        self.assertEqual(original, before)

    def test_labels_and_unknown_fields_cannot_enter_query(self):
        for field in ('answer_index', 'gt', 'admission_group', 'frames', 'unknown'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_queries([dict(query(2), **{field: 0})])

    def test_invalid_times_paths_and_options_are_rejected(self):
        for value in (True, float('nan'), float('inf'), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_queries([query(value)])
        for path in ('../other', '/absolute', 'D:/other', 'a\\b', 'a//b', './a'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_queries([query(2, source=path)])
        for options in ([], ['only'], ['a', ''], 'ab'):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_queries([dict(query(2), options=options)])

    def test_duplicate_query_id_is_rejected_across_sources(self):
        with self.assertRaises(ValueError):
            validate_queries([query(2), query(3, source='other.mp4')])

    def test_verified_label_requires_exact_question_options_time_and_source(self):
        request = query(2, 'ovo:1')
        raw = dict(id=1, video='video.mp4', realtime=2, question=request['question'],
                   options=request['options'], gt=1, answer='second')
        self.assertEqual(bind_official_labels([request], [dict(request_id='ovo:1', record=raw)]), {'ovo:1': 1})
        for field, value in (('video', 'other.mp4'), ('realtime', 3), ('question', 'other'),
                             ('options', ['second', 'first']), ('gt', True), ('answer', 'first')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind_official_labels([request], [dict(request_id='ovo:1', record=dict(raw, **{field: value}))])

    def test_missing_labels_and_unverified_admission_never_become_accuracy(self):
        result = pending_catalog([query(2, 'a'), query(3, 'b')], {'a': 0})
        self.assertEqual([r['original_label_index'] for r in result], [0, None])
        self.assertTrue(all(r['admission_group'] == 'pending' and not r['eligible_for_confirmed_accuracy']
                            and r['human_semantic_review'] == 'pending' for r in result))


class ContinuousContracts(unittest.TestCase):
    def test_query_pauses_without_requesting_future_sample(self):
        source, observed = Samples(specs(12)), []
        def ask(q, state):
            self.assertEqual(source.index, q['query_time_s']+1)
            observed.append(state)
        result = replay_source(specs(12), [query(2), query(6, 'later')], source, lambda *args: None, ask)
        self.assertEqual(result['encoded_pairs'], 3)
        self.assertEqual([s['committed_pairs'] for s in observed], [1, 3])
        self.assertEqual(source.index, 7)
        self.assertTrue(source.closed)

    def test_unmatched_frame_survives_query_and_pairs_with_next_real_frame(self):
        pairs, seen = [], []
        replay_source(specs(8), [query(2), query(3, 'next')], Samples(specs(8)),
                      lambda i, a, b: pairs.append((i, a.frame_id, b.frame_id)), lambda q, s: seen.append(s))
        self.assertEqual(pairs, [(0, 0, 1), (1, 2, 3)])
        self.assertEqual(seen[0]['pending_frame']['frame_id'], 2)
        self.assertIsNone(seen[1]['pending_frame'])
        self.assertFalse(any(s['query_padding'] for s in seen))

    def test_same_time_queries_share_identical_bank_prefix_without_decoding(self):
        source, seen = Samples(specs(8)), []
        replay_source(specs(8), [query(4, 'a'), query(4, 'b')], source, lambda *args: None,
                      lambda q, s: seen.append((source.index, s)))
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(source.index, 5)

    def test_query_exactly_at_sample_time_includes_that_sample_only(self):
        seen = []
        replay_source(specs(8), [query(2.999, 'a'), query(3, 'b')], Samples(specs(8)),
                      lambda *args: None, lambda q, s: seen.append(s['arrived_frames']))
        self.assertEqual(seen, [3, 4])

    def test_final_odd_frame_is_not_padded_or_encoded(self):
        pairs = []
        result = replay_source(specs(5), [query(99)], Samples(specs(5)),
                               lambda i, a, b: pairs.append((a.frame_id, b.frame_id)), lambda *args: None)
        self.assertEqual(pairs, [(0, 1), (2, 3)])
        self.assertEqual(result['unmatched_final_frame'], 1)

    def test_prefix_hashes_are_identical_for_unchanged_prefix_and_distinguish_pending(self):
        seen = []
        replay_source(specs(8), [query(1, 'a'), query(2, 'b'), query(2, 'c')], Samples(specs(8)),
                      lambda *args: None, lambda q, s: seen.append(s))
        self.assertEqual(seen[0]['committed_frames_sha256'], seen[1]['committed_frames_sha256'])
        self.assertNotEqual(seen[0]['arrived_frames_sha256'], seen[1]['arrived_frames_sha256'])
        self.assertEqual(seen[1], seen[2])

    def test_historical_hash_matches_original_frame_list_serialization(self):
        frames, observed = specs(8), []
        replay_source(frames, [query(4)], Samples(frames), lambda *args: None,
                      lambda q, s: observed.append(s))
        expected = hashlib.sha256(json.dumps(frames[:4], sort_keys=True).encode()).hexdigest()
        self.assertEqual(observed[0]['historical_committed_frames_sha256'], expected)
        self.assertEqual(observed[0]['committed_through_s'], 3.)
        self.assertEqual(observed[0]['observation_age_s'], 1.)

    def test_input_queries_and_frame_specs_cannot_be_mutated_by_query_callback(self):
        frames, queries = specs(6), [query(2)]
        before = copy.deepcopy((frames, queries))
        def mutate(q, state):
            q['options'][0] = 'changed'
            state['pending_frame']['frame_id'] = 1000
        replay_source(frames, queries, Samples(frames), lambda *args: None, mutate)
        self.assertEqual((frames, queries), before)

    def test_early_eof_sample_identity_and_future_packet_are_rejected(self):
        for frames in (specs(2), [dict(f, frame_id=f['frame_id']+1) for f in specs(6)],
                       [dict(f, pts_s=f['pts_s']+10) for f in specs(6)]):
            source = Samples(frames)
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                replay_source(specs(6), [query(5)], source, lambda *args: None, lambda *args: None)
            self.assertTrue(source.closed)

    def test_readout_and_pair_exceptions_stop_without_future_pull_and_close_source(self):
        def fail(*args):
            raise RuntimeError('injected failure')
        for pair_callback, query_callback, expected_count in ((fail, lambda *args: None, 2),
                                                              (lambda *args: None, fail, 3)):
            source = Samples(specs(8))
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                replay_source(specs(8), [query(2), query(7, 'b')], source, pair_callback, query_callback)
            self.assertTrue(source.closed)
            self.assertEqual(source.index, expected_count)

    def test_invalid_frame_specs_and_empty_pair_query_fail_before_sample_pull(self):
        for frame_list in ([], [dict(specs(1)[0], pts_s=-.1)],
                           [dict(specs(1)[0], frame_id=True)], [dict(specs(1)[0], rgb_sha256='not SHA')],
                           [specs(1)[0], specs(1)[0]]):
            with self.subTest(frame_list=frame_list), self.assertRaises(ValueError):
                validate_frames(frame_list)
        source = Samples(specs(8))
        with self.assertRaises(ValueError):
            replay_source(specs(8), [query(0)], source, lambda *args: None, lambda *args: None)
        self.assertEqual(source.index, 0)


class FakeContainer:
    def __init__(self, frames):
        self.frames, self.pulled, self.closed = frames, [], False
        self.streams = SimpleNamespace(video=[SimpleNamespace(codec_context=SimpleNamespace(thread_count=0))])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def decode(self, stream):
        for i, frame in enumerate(self.frames):
            self.pulled.append(i)
            yield frame


class DecoderContracts(unittest.TestCase):
    def environment(self, count=8):
        raw = [SimpleNamespace(time=float(i)+10, data=bytes([i])*192*256*3) for i in range(count)]
        container = FakeContainer(raw)
        frames = [dict(frame_id=i, pts_s=float(i), rgb_sha256=hashlib.sha256(f.data).hexdigest()) for i, f in enumerate(raw)]
        env = dict(av=SimpleNamespace(open=lambda p: container), np=np, canvas_bytes=lambda f: f.data,
                   FramePacket=lambda i, pts, rgb, n: SimpleNamespace(frame_id=i, timestamp=pts, rgb=rgb))
        return env, container, frames

    def test_real_decoder_adapter_does_not_decode_past_paused_query(self):
        env, container, frames = self.environment()
        replay_source(frames, [query(2)], decode_samples(env, 'fake.mp4', frames), lambda *args: None,
                      lambda q, s: self.assertEqual(container.pulled, [0, 1, 2]))
        self.assertTrue(container.closed)
        self.assertEqual(container.streams.video[0].codec_context.thread_count, 1)

    def test_decoder_uses_actual_raw_ids_and_normalized_pts(self):
        env, container, frames = self.environment()
        selected = [frames[i] for i in (0, 2, 4)]
        packets = list(decode_samples(env, 'fake.mp4', selected))
        self.assertEqual([(p.frame_id, p.timestamp) for p in packets], [(0, 0.), (2, 2.), (4, 4.)])
        self.assertEqual(container.pulled, [0, 1, 2, 3, 4])

    def test_rgb_and_pts_drift_are_rejected_and_container_closed(self):
        for field, value in (('rgb_sha256', '0'*64), ('pts_s', .1)):
            env, container, frames = self.environment()
            frames[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                list(decode_samples(env, 'fake.mp4', frames))
            self.assertTrue(container.closed)

    def test_missing_or_repeated_pts_are_rejected(self):
        for value in (None, 10.):
            env, container, frames = self.environment()
            container.frames[1].time = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(decode_samples(env, 'fake.mp4', frames))
            self.assertTrue(container.closed)

    def test_early_media_eof_is_rejected(self):
        env, container, frames = self.environment()
        container.frames = container.frames[:2]
        with self.assertRaises(ValueError):
            list(decode_samples(env, 'fake.mp4', frames))
        self.assertTrue(container.closed)


class SessionIntegrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def encode(self, index, first, second):
        tensor = torch.randn(48, 32, generator=torch.Generator().manual_seed(800+index))
        return policies().features.raw_block(tensor, index, first.timestamp, second.timestamp)

    def test_six_sessions_match_independent_uninterrupted_prefixes_with_single_encoding(self):
        frames, queries = specs(40), [query(8, 'a'), query(17, 'b'), query(39, 'c')]
        references, source = {}, Samples(frames)
        for arm in ARMS[:-1]:
            session = MemorySession(arm)
            for i in range(20):
                a, b = (SimpleNamespace(frame_id=j, timestamp=float(j)) for j in (2*i, 2*i+1))
                session.append_pair(self.encode(i, a, b), frame_ids=(a.frame_id, b.frame_id))
                references[arm, i+1] = digest(session.export())
            session.close()
        answers, encoded = [], []
        def encoder(*args):
            encoded.append(args[0])
            return self.encode(*args)
        def readout(nodes, text):
            return dict(text='A', generated_ids=[32, 1], observed_digest=digest(nodes))
        prompt = module('score_memory_eval', NATURAL).prompt
        receipt = run_sessions(frames, queries, source, encoder, readout, prompt, lambda e: None, answers.append)
        self.assertEqual(encoded, list(range(20)))
        self.assertEqual(receipt['answers'], 18)
        self.assertEqual(receipt['reset_checks'], 5)
        for answer in answers:
            count = answer['prefix']['committed_pairs']
            if answer['arm'] != 'text_only':
                self.assertEqual(answer['observed_digest'], references[answer['arm'], count])
                self.assertTrue(answer['memory_unchanged'])
            else:
                self.assertEqual(answer['observed_digest'], digest([]))
        self.assertTrue(source.closed)

    def test_readout_failure_closes_all_sessions_without_continuing_decode(self):
        closed = []
        original = MemorySession.close
        def close(session):
            closed.append(session.policy)
            original(session)
        def fail(*args):
            raise RuntimeError('readout failed')
        source = Samples(specs(20))
        with patch.object(MemorySession, 'close', close), self.assertRaisesRegex(RuntimeError, 'readout failed'):
            run_sessions(specs(20), [query(3), query(19, 'b')], source, self.encode, fail,
                         lambda q: q['question'], lambda e: None, lambda a: None)
        self.assertEqual(set(closed), set(ARMS[:-1]))
        self.assertEqual(source.index, 4)
        self.assertTrue(source.closed)

    def test_answer_budget_violation_is_not_silently_truncated(self):
        with self.assertRaisesRegex(RuntimeError, 'generation budget'):
            run_sessions(specs(6), [query(5)], Samples(specs(6)), self.encode,
                         lambda *args: dict(text='A', generated_ids=[1]*9),
                         lambda q: q['question'], lambda e: None, lambda a: None)

    def test_prompt_contains_only_original_question_and_options(self):
        captured = []
        def prompt(value):
            captured.append(value)
            return 'question'
        run_sessions(specs(4), [query(3)], Samples(specs(4)), self.encode,
                     lambda *args: dict(text='A', generated_ids=[32]), prompt, lambda e: None, lambda a: None)
        self.assertEqual(captured, [dict(question='Which option?', options=['first', 'second'])])


if __name__ == '__main__':
    unittest.main()
