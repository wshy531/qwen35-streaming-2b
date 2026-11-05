"""Non-model checks that would catch misleading memory evaluation conclusions."""
import gc
import json
import math
from pathlib import Path
import time
import unittest
import weakref

from audit_sampling import contains_all, query_snapshot
from memory_policy import Pair, RecentFIFO, RecentLongHierarchy
from score_memory_eval import paired, score, parse_choice


class PreparationTests(unittest.TestCase):
    def test_denominators_and_video_balance(self):
        rows = [dict(id=str(i),dataset='x',source_id='a' if i<3 else 'b',
            task='historical',options=['yes','no'],answer_index=0) for i in range(4)]
        result = score(rows,[dict(id='0',text='A'),dict(id='1',text='A because yes'),
                             dict(id='2',status='timeout')])
        self.assertEqual(result['overall']['micro_accuracy'],.25)
        self.assertAlmostEqual(result['overall']['video_macro_accuracy'],1/6)
        self.assertEqual(result['overall']['answered_only_accuracy'],1)
        self.assertEqual(result['overall']['statuses'],dict(ok=1,format_error=1,timeout=1,missing=1))
        candidate = score(rows,[dict(id='3',text='A')])
        self.assertEqual(paired(result,candidate),dict(new_loss=1,both_wrong=2,correction=1))
        with self.assertRaises(ValueError):
            score(rows,[dict(id='0',text='A'),dict(id='0',text='B')])
        with self.assertRaises(ValueError):
            score(rows,[dict(id='unknown',text='A')])

    def test_unanswerable_is_not_assumed_first_option(self):
        rows = [dict(id='n',dataset='x',source_id='b',task='HLD',
                     options=['left','right','Unable to answer'],answer_index=2)]
        self.assertEqual(score(rows,[dict(id='n',text='C')])['overall']['correct'],1)
        self.assertIsNone(parse_choice('D',3))
        self.assertIsNone(parse_choice('option C',3))

    def test_all_evidence_and_no_future_or_tail_leak(self):
        self.assertFalse(contains_all([[1,2],[10,11]],[1.5]))
        frames = [dict(frame_id=i,pts_s=float(i)) for i in range(7)]
        early = query_snapshot(frames,2.,[[1,2]])
        same = query_snapshot(frames[:3],2.,[[1,2]])
        self.assertEqual(early,same)
        self.assertEqual(early['M0']['real_frame_ids'],[0,1,2])
        self.assertTrue(early['odd_tail_padding'])
        self.assertEqual(query_snapshot(frames,5.,None)['M0']['real_frame_ids'],list(range(6)))

    def test_bounded_storage_eviction_and_prefix(self):
        class Payload:
            pass
        policy = RecentLongHierarchy(4,4)
        refs = weakref.WeakValueDictionary()
        t0 = time.monotonic()
        prefix = None
        for i in range(100_000):
            payload = Payload()
            payload.data = bytes(1024)
            refs[i] = payload
            policy.append(Pair(i,float(i*4),float(i*4+2),payload))
            if i in (15,999,99_999):
                gc.collect()
                self.assertLessEqual(len(refs),8)
                self.assertLessEqual(len(policy.read()),8)
                self.assertLessEqual(len(policy.long),4)
            if i == 15:
                prefix = tuple(p.index for p in policy.read())
        self.assertLess(time.monotonic()-t0,120)
        other = RecentLongHierarchy(4,4)
        for i in range(16):
            other.append(Pair(i,float(i*4),float(i*4+2)))
        self.assertEqual(prefix,tuple(p.index for p in other.read()))
        before = other.provenance(),other.read()
        other.read()
        self.assertEqual(before,(other.provenance(),other.read()))
        with self.assertRaises(ValueError):
            other.append(Pair(15,60.,62.))
        with self.assertRaises(ValueError):
            other.append(Pair(16,float('nan'),66.))
        self.assertEqual(before,(other.provenance(),other.read()))
        self.assertEqual([p.index for p in policy.recent],list(range(99_996,100_000)))
        # first representative survives, but its span does not mean all past pairs survive
        self.assertEqual(policy.read()[0].index,0)
        self.assertNotIn(1,[p.index for p in policy.read()])

    def test_same_payload_budget_different_selection(self):
        fifo, hierarchy = RecentFIFO(8),RecentLongHierarchy(4,4)
        for i in range(40):
            pair = Pair(i,i*4.,i*4.+2.)
            fifo.append(pair)
            hierarchy.append(pair)
        self.assertEqual(len(fifo.read()),8)
        self.assertEqual(len(hierarchy.read()),8)
        self.assertEqual([p.index for p in fifo.read()],list(range(32,40)))
        self.assertEqual([p.index for p in hierarchy.read()][-4:],list(range(36,40)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
