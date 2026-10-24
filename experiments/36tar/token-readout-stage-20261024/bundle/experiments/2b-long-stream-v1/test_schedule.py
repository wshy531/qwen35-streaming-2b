"""Small CPU contracts for the failure modes that could leak future frames."""
import unittest

from schedule import Frame,Query,events


class CausalSchedule(unittest.TestCase):
    def run_schedule(self,qs,n=6):
        return list(events((Frame(i,float(i)) for i in range(n)),qs,end_s=float(n)))

    def test_odd_pending_does_not_use_next_frame(self):
        xs=self.run_schedule([Query('q',2.5)])
        q=next(x for x in xs if x['kind']=='query')
        self.assertEqual(q['committed_through_s'],1.)
        self.assertEqual(q['pending_frame_id'],2)
        self.assertLess(xs.index(q),next(i for i,x in enumerate(xs) if x['kind']=='receive' and x['frame_id']==3))

    def test_continue_after_two_questions_without_replay(self):
        xs=self.run_schedule([Query('a',1.5),Query('b',3.5)])
        self.assertEqual([x['frame_ids'] for x in xs if x['kind']=='pair'],[[0,1],[2,3],[4,5]])
        self.assertEqual([x['committed_pairs'] for x in xs if x['kind']=='query'],[1,2])

    def test_frame_at_exact_query_time_precedes_question(self):
        q=next(x for x in self.run_schedule([Query('q',1.)]) if x['kind']=='query')
        self.assertEqual(q['committed_through_s'],1.)

    def test_same_timestamp_questions_share_observation(self):
        qs=[x for x in self.run_schedule([Query('a',2.),Query('b',2.)]) if x['kind']=='query']
        self.assertEqual(qs[0]['committed_pairs'],qs[1]['committed_pairs'])
        self.assertEqual(qs[0]['pending_frame_id'],qs[1]['pending_frame_id'])

    def test_initial_question_and_eof_tail_are_explicit(self):
        xs=self.run_schedule([Query('a',0.),Query('b',2.9)],n=3)
        qs=[x for x in xs if x['kind']=='query']
        self.assertTrue(qs[0]['no_visible_pair'])
        self.assertEqual(qs[1]['pending_frame_id'],2)
        self.assertFalse(xs[-1]['incomplete_tail_committed'])

    def test_bad_time_or_order_is_rejected(self):
        for frames in [[Frame(0,1.),Frame(1,0.)],[Frame(0,float('nan'))],[Frame(0,9.)]]:
            with self.assertRaises(ValueError):list(events(frames,[],end_s=3.))
        for qs in [[Query('q',4.)],[Query('q',1.),Query('q',2.)],[Query('b',2.),Query('a',1.)],
                   ['invalid'],[Query('q',float('nan'))],[Query('q',True)]]:
            with self.assertRaises(ValueError):self.run_schedule(qs,n=3)

    def test_lazy_source_is_not_materialized(self):
        consumed=[]
        def source():
            for i in range(10000):
                consumed.append(i);yield Frame(i,float(i))
        it=events(source(),[Query('q',1.5)],end_s=10000.)
        for x in it:
            if x['kind']=='query':break
        self.assertEqual(consumed,[0,1,2])
        self.assertEqual(x['committed_through_s'],1.)


if __name__=='__main__':unittest.main()
