import unittest
from experiments.wshy.sember_resolution_2b.resources import forecast


class Resources(unittest.TestCase):
    def data(self,count=195):
        return ([dict(id='q',arm='B',shift=0,input_tokens=1999+48*count),
                 dict(id='q',arm='R',shift=0,input_tokens=1999+108*count)],
                [dict(id='q',B=[{}]*(count*2))])
    def test_long_evidence_budget(self):
        report=forecast(*self.data())
        self.assertEqual(report['maximum_input_tokens'],39439)
        self.assertEqual(report['questions_over_old_24K'],1)
        self.assertGreater(report['workspace']['two_full_attention_matrices_GiB'],72)
        self.assertLess(report['workspace']['one_tiled_attention_matrix_GiB'],1)
    def test_reject_unexplained_token_delta(self):
        rows,selections=self.data(); rows[1]['input_tokens']+=1
        with self.assertRaises(ValueError): forecast(rows,selections)
    def test_reject_context_overflow(self):
        with self.assertRaises(ValueError): forecast(*self.data(250))
