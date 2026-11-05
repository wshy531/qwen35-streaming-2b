"""Tests for query snapshots corrupting later real-frame pairing/history."""
import copy
import math
import unittest

from audit_sampling import query_snapshot
from memory_query_blueprint import MemoryQueryBlueprint


def frames(n=61):
    return [dict(frame_id=i,pts_s=i*2.,rgb_sha256=f'{i:064x}') for i in range(n)]


class MemoryQueryBlueprintTests(unittest.TestCase):
    def test_odd_query_tail_does_not_commit_and_next_real_frame_pairs_normally(self):
        bp=MemoryQueryBlueprint(frames())
        snap=bp.query(4.)
        self.assertEqual(snap['committed_real_pairs'],1)
        self.assertEqual(snap['pending_frame_id'],2)
        self.assertTrue(snap['M0'][-1]['payload']['snapshot_padding'])
        self.assertEqual(bp.state()['M0'][-1]['payload']['frames'][-1]['frame_id'],1)
        next_snap=bp.query(6.)
        self.assertEqual([f['frame_id'] for f in next_snap['M0'][-1]['payload']['frames']],[2,3])
        self.assertFalse(next_snap['M0'][-1]['payload']['snapshot_padding'])

    def test_extra_questions_cannot_change_later_persistent_memories(self):
        a=MemoryQueryBlueprint(frames());b=MemoryQueryBlueprint(frames())
        for t in (0,4,10,18,22,38,42,70):a.query(t)
        a.query(100.);b.query(100.)
        self.assertEqual(a.state(),b.state())

    def test_future_frames_are_ignored_by_current_query(self):
        a=frames();b=copy.deepcopy(a)
        for f in b:
            if f['pts_s']>20:f['rgb_sha256']='f'*64
        self.assertEqual(MemoryQueryBlueprint(a).query(20.),MemoryQueryBlueprint(b).query(20.))

    def test_online_forks_match_fresh_prefix_for_every_query(self):
        f=frames();bp=MemoryQueryBlueprint(f)
        for query in range(121):
            actual=bp.query(query);expected=query_snapshot(f,query,None)
            for name in ('M0','M1'):
                self.assertEqual([p['index'] for p in actual[name]],expected[name]['pair_indices'])

    def test_out_of_order_query_cannot_rewind_a_stream(self):
        bp=MemoryQueryBlueprint(frames());bp.query(20.)
        with self.assertRaises(ValueError):bp.query(10.)

    def test_nonfinite_cutoff_cannot_disable_causality_checks(self):
        bp=MemoryQueryBlueprint(frames());bp.query(20.);original=bp.state()
        for invalid in (math.nan,math.inf,-1,True):
            with self.assertRaises(ValueError):bp.query(invalid)
            self.assertEqual(bp.state(),original)


if __name__=='__main__':unittest.main(verbosity=2)
