"""Must run with real torch. Missing torch is an error, never a passing skip."""
import unittest
import torch
from types import SimpleNamespace
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from experiments.wshy.sember_resolution_2b.native import tiled_attention,query_tiling
from experiments.wshy.sember_resolution_2b.memory import Bank,raw,merge,validate
from experiments.wshy.sember_resolution_2b.checks import low_parity
from experiments.wshy.sember_resolution_2b.controls import PRESETS,geometry


class Numeric(unittest.TestCase):
    def attention(self,qlen=19,klen=19):
        generator=torch.Generator().manual_seed(32)
        return (SimpleNamespace(training=False,is_causal=True,num_key_value_groups=2),
                torch.randn(1,4,qlen,16,generator=generator),
                torch.randn(1,2,klen,16,generator=generator),
                torch.randn(1,2,klen,16,generator=generator))
    def test_tiled_attention_causal_and_masks(self):
        module,q,k,v=self.attention()
        causal=torch.ones(19,19,dtype=torch.bool).tril()
        floating=torch.zeros(19,19).masked_fill(~causal,float('-inf'))
        for mask,causal_flag in ((None,True),(causal,True),(floating,True),(None,False)):
            with sdpa_kernel(SDPBackend.MATH):
                expected,_=sdpa_attention_forward(module,q,k,v,mask,scaling=.25,is_causal=causal_flag)
                actual,_=tiled_attention(sdpa_attention_forward,module,q,k,v,mask,
                    scaling=.25,is_causal=causal_flag,tile=4)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    def test_tiled_attention_cached_explicit_mask(self):
        module,q,k,v=self.attention(19,23)
        mask=torch.arange(23)[None,:]<=torch.arange(4,23)[:,None]
        with sdpa_kernel(SDPBackend.MATH):
            expected,_=sdpa_attention_forward(module,q,k,v,mask,scaling=.25)
            actual,_=tiled_attention(sdpa_attention_forward,module,q,k,v,mask,scaling=.25,tile=4)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        with self.assertRaises(ValueError):
            tiled_attention(sdpa_attention_forward,module,q,k,v,None,tile=4)
    def test_tiled_attention_no_future_leak(self):
        module,q,k,v=self.attention()
        with sdpa_kernel(SDPBackend.MATH):
            expected,_=tiled_attention(sdpa_attention_forward,module,q,k,v,None,tile=4)
            k[:,:,8:]+=100; v[:,:,8:]-=100
            actual,_=tiled_attention(sdpa_attention_forward,module,q,k,v,None,tile=4)
        torch.testing.assert_close(actual[:,:8],expected[:,:8],rtol=0,atol=0)
    def test_single_token_decode_is_original(self):
        module,q,k,v=self.attention(1,23)
        with sdpa_kernel(SDPBackend.MATH):
            expected,_=sdpa_attention_forward(module,q,k,v,None)
            actual,_=tiled_attention(sdpa_attention_forward,module,q,k,v,None,tile=4)
        self.assertTrue(torch.equal(actual,expected))
    def test_attention_dispatch_restored_on_failure(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        class Qwen3_5Attention(torch.nn.Module): pass
        model=torch.nn.Sequential(Qwen3_5Attention())
        before=ALL_ATTENTION_FUNCTIONS['sdpa']
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with query_tiling(model):
                self.assertIsNot(ALL_ATTENTION_FUNCTIONS['sdpa'],before)
                raise RuntimeError('deliberate')
        self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],before)
    def node(self,r,i=0):
        return raw(torch.arange(geometry(r)['pair_tokens']*32,dtype=torch.float32).reshape(-1,32),i,2.*i,2.*i+1,r)
    def test_low_bitwise_equivalence(self):
        low_parity(torch)
    def test_all_grids_conserve_mass(self):
        for r in PRESETS:
            parent=merge(self.node(r),self.node(r,1))
            self.assertEqual(len(parent.features),geometry(r)['pair_tokens'])
            self.assertEqual(float(parent.mass.sum()),2*geometry(r)['pair_tokens'])
            validate(parent)
    def test_grid_mismatch_rejected(self):
        with self.assertRaises(ValueError): merge(self.node('low'),self.node('mid',1))
    def test_repeated_pair_rejected_without_mutation(self):
        bank=Bank('LS','high'); n=self.node('high'); bank.append(n); before=bank.digest()
        with self.assertRaises(ValueError): bank.append(n)
        self.assertEqual(bank.digest(),before)
    def test_owned_features(self):
        bank=Bank('A06','mid'); n=self.node('mid'); bank.append(n); before=bank.digest()
        n.features.zero_(); self.assertEqual(bank.digest(),before)
    def test_large_stream_bounds_and_reset(self):
        torch.manual_seed(9)
        for r in PRESETS:
            for arm in ('LS','A06'):
                bank=Bank(arm,r)
                for i in range(160):
                    bank.append(raw(torch.randn(geometry(r)['pair_tokens'],32),i,2.*i,2.*i+1,r))
                    self.assertLessEqual(bank.status()['tokens'],geometry(r)['total_tokens'])
                    self.assertLessEqual(len(bank.nodes()),8)
                self.assertEqual(len(bank.recent),4)
                bank.reset(); self.assertEqual(bank.digest(),Bank(arm,r).digest())
                self.assertEqual(bank.status()['resident_bytes'],0)


if __name__=='__main__': unittest.main()
