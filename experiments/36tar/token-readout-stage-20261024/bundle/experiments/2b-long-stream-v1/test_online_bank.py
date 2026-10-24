"""Mechanism tests only; synthetic features do not measure visual accuracy."""
import sys
import unittest

from online_bank import BASE,OnlineBank,nodes_digest
import torch

sys.path.insert(0,str(BASE))
import fixed_stage2
fixed_stage2.DIAG=BASE/'diag'
from legacy_sources import load
from forest import raw
from features import raw_block


class IncrementalBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_same_frozen_policy_at_multiple_queries(self):
        generator=torch.Generator().manual_seed(20261020)
        history=[];banks={k:OnlineBank(k) for k in ('M0','M2')}
        for i in range(240):
            # Structured segments plus noise exercise within/cross merges.
            x=torch.randn((48,32),generator=generator)*.03+(i//25+1)
            history.append(raw(raw_block(x,i,float(2*i),float(2*i+1))))
            for b in banks.values():b.append(x,i,float(2*i),float(2*i+1))
            if i in (7,63,149,239):
                expected,_,_=fixed_stage2.policies({'load':load},history)
                for name,b in banks.items():
                    self.assertEqual(b.fingerprint(),nodes_digest(expected[name]))
                    self.assertLessEqual(b.status()['visual_tokens'],384)

    def test_query_exports_do_not_mutate_bank_then_input_continues(self):
        for name in ('M0','M2'):
            b=OnlineBank(name)
            for i in range(10):b.append(torch.ones((48,32)),i,2.*i,2.*i+1)
            before=b.fingerprint();view=b.export();view[0].features.zero_()
            self.assertEqual(before,b.fingerprint())
            b.append(torch.ones((48,32)),10,20.,21.)
            self.assertEqual(b.status()['processed_pairs'],11)

    def test_no_evicted_source_payload_survives_in_recent_window(self):
        import weakref,gc
        b=OnlineBank('M0');old=None
        for i in range(20):
            b.append(torch.ones((48,32)),i,2.*i,2.*i+1)
            if i==0:old=weakref.ref(b.bank.read()[0].payload)
        gc.collect()
        self.assertIsNone(old())
        self.assertEqual([n.first for n in b.export()],list(range(12,20)))

    def test_duplicate_and_out_of_order_input_rejected(self):
        for name in ('M0','M2'):
            b=OnlineBank(name);b.append(torch.ones((48,32)),0,0.,1.)
            before=b.fingerprint()
            for index,begin,end in [(0,2.,3.),(2,2.,3.),(1,1.,2.)]:
                with self.assertRaises(ValueError):b.append(torch.ones((48,32)),index,begin,end)
                self.assertEqual(b.fingerprint(),before)


if __name__=='__main__':unittest.main()
